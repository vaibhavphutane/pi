import { anthropicMessagesApi } from "../api/anthropic-messages.lazy.ts";
import { openAICompletionsApi } from "../api/openai-completions.lazy.ts";
import { openrouterImagesApi } from "../api/openrouter-images.lazy.ts";
import { typesafeSystemOneApi } from "../api/typesafe-system-one.lazy.ts";
import { envApiKeyAuth, lazyOAuth } from "../auth/helpers.ts";
import { loadOpenRouterOAuth } from "../auth/oauth/load.ts";
import type { Credential } from "../auth/types.ts";
import { createProvider, type Provider } from "../models.ts";
import { raceWithAbortSignal } from "../utils/abort.ts";
import { OPENROUTER_CLASSIFIER_MODELS, OPENROUTER_IMAGE_MODELS, OPENROUTER_MODELS } from "./openrouter.models.ts";

const USER_MODELS_URL = "https://openrouter.ai/api/v1/models/user?output_modalities=all";
const CATALOG_TIMEOUT_MS = 4_000;
const CATALOG_REFRESH_INTERVAL_MS = 5 * 60 * 1000;

function apiKey(credential: Credential | undefined): string | undefined {
	return credential?.type === "oauth" ? credential.access : credential?.key;
}

async function credentialScope(key: string): Promise<string> {
	const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(key));
	return `openrouter:${Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

function parseApprovedModelIds(value: unknown): string[] {
	if (typeof value !== "object" || value === null || !("data" in value) || !Array.isArray(value.data)) {
		throw new Error("OpenRouter approved model catalog returned invalid data");
	}
	const ids = new Set<string>();
	for (const entry of value.data) {
		if (
			typeof entry !== "object" ||
			entry === null ||
			!("id" in entry) ||
			typeof entry.id !== "string" ||
			!entry.id
		) {
			throw new Error("OpenRouter approved model catalog contains an invalid model ID");
		}
		ids.add(entry.id);
	}
	return [...ids];
}

export function openrouterProvider(): Provider<"anthropic-messages" | "openai-completions"> {
	let approvedKey: string | undefined;
	let approvedIds = new Set<string>();
	let checkedAt: number | undefined;
	const invalidatedScopes = new Set<string>();
	const provider = createProvider<"anthropic-messages" | "openai-completions">({
		id: "openrouter",
		name: "OpenRouter",
		baseUrl: "https://openrouter.ai/api/v1",
		auth: {
			apiKey: envApiKeyAuth("OpenRouter API key", ["OPENROUTER_API_KEY"]),
			oauth: lazyOAuth({
				name: "OpenRouter OAuth",
				loginLabel: "Sign in with OpenRouter",
				load: loadOpenRouterOAuth,
			}),
		},
		models: [
			...Object.values(OPENROUTER_MODELS),
			...Object.values(OPENROUTER_IMAGE_MODELS),
			...Object.values(OPENROUTER_CLASSIFIER_MODELS),
		],
		filterModels: (models, credential) =>
			apiKey(credential) && apiKey(credential) === approvedKey
				? models.filter((model) => approvedIds.has(model.id))
				: [],
		filterAllModels: (models, credential) =>
			apiKey(credential) && apiKey(credential) === approvedKey
				? models.filter((model) => approvedIds.has(model.id))
				: [],
		api: {
			"anthropic-messages": anthropicMessagesApi(),
			"openai-completions": openAICompletionsApi(),
		},
		images: { "openrouter-images": openrouterImagesApi() },
		// OpenRouter serves TypeSafe's System One protocol at /api/v1/systemone.
		classifiers: { "typesafe-system-one": typesafeSystemOneApi() },
	});
	return {
		...provider,
		credentialScopedModels: true,
		refreshModels: async (context) => {
			const key = apiKey(context.credential);
			const scope = key ? await credentialScope(key) : undefined;
			const stored = context.stored;
			// Public catalogs and another workspace's approvals are never a fallback.
			const cachedIds =
				scope &&
				!invalidatedScopes.has(scope) &&
				stored?.credentialScope === scope &&
				Array.isArray(stored.approvedModelIds)
					? stored.approvedModelIds.filter((id): id is string => typeof id === "string" && id.length > 0)
					: undefined;
			if (
				!(await context.publish({
					update: () => {
						if (
							cachedIds &&
							(approvedKey !== key || checkedAt === undefined || (stored?.checkedAt ?? 0) > checkedAt)
						) {
							approvedKey = key;
							approvedIds = new Set(cachedIds);
							checkedAt = stored?.checkedAt;
						} else if (!key || approvedKey !== key) {
							approvedKey = undefined;
							approvedIds = new Set();
							checkedAt = undefined;
						}
					},
				}))
			) {
				return;
			}
			if (!key || !context.allowNetwork || context.signal.aborted) return;
			if (
				cachedIds &&
				!context.force &&
				stored?.checkedAt !== undefined &&
				Date.now() - stored.checkedAt < CATALOG_REFRESH_INTERVAL_MS
			) {
				return;
			}

			const signal = AbortSignal.any([context.signal, AbortSignal.timeout(CATALOG_TIMEOUT_MS)]);
			try {
				const response = await raceWithAbortSignal(
					fetch(USER_MODELS_URL, {
						headers: { accept: "application/json", authorization: `Bearer ${key}` },
						signal,
					}),
					signal,
				);
				if (!response.ok) {
					if (response.status === 401 || response.status === 403) {
						await context.publish({
							update: () => {
								if (scope) invalidatedScopes.add(scope);
								approvedKey = undefined;
								approvedIds = new Set();
								checkedAt = undefined;
							},
						});
						await context.publish({ persist: null });
					}
					throw new Error(`OpenRouter approved model catalog request failed (HTTP ${response.status})`);
				}
				const ids = parseApprovedModelIds(await raceWithAbortSignal(response.json(), signal));
				signal.throwIfAborted();
				const allowed = new Set(ids);
				const verifiedAt = Date.now();
				// Apply authoritative approvals even if the cache cannot be written.
				if (
					!(await context.publish({
						update: () => {
							approvedKey = key;
							approvedIds = allowed;
							checkedAt = verifiedAt;
							if (scope) invalidatedScopes.add(scope);
						},
					}))
				) {
					return;
				}
				await context.publish({
					persist: {
						models: (provider.getAllModels?.() ?? provider.getModels()).filter((model) => allowed.has(model.id)),
						credentialScope: scope,
						approvedModelIds: ids,
						checkedAt: verifiedAt,
					},
					update: () => {
						if (scope) invalidatedScopes.delete(scope);
					},
				});
			} catch (error) {
				context.signal.throwIfAborted();
				if (signal.aborted) throw new Error("OpenRouter approved model catalog request timed out");
				if (error instanceof Error && error.message.startsWith("OpenRouter approved model catalog")) throw error;
				throw new Error("Could not refresh OpenRouter approved model catalog", { cause: error });
			}
		},
	};
}
