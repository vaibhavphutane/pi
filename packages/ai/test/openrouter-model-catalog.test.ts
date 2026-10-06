import { afterEach, describe, expect, it, vi } from "vitest";
import { InMemoryCredentialStore } from "../src/auth/credential-store.ts";
import type { AuthContext, Credential } from "../src/auth/types.ts";
import { createModels } from "../src/models.ts";
import { InMemoryModelsStore } from "../src/models-store.ts";
import {
	OPENROUTER_CLASSIFIER_MODELS,
	OPENROUTER_IMAGE_MODELS,
	OPENROUTER_MODELS,
} from "../src/providers/openrouter.models.ts";
import { openrouterProvider } from "../src/providers/openrouter.ts";

const chat = Object.values(OPENROUTER_MODELS)[0];
const otherChat = Object.values(OPENROUTER_MODELS)[1];
const image = Object.values(OPENROUTER_IMAGE_MODELS)[0];
const classifier = Object.values(OPENROUTER_CLASSIFIER_MODELS)[0];

function catalogResponse(ids: readonly string[]): Response {
	return Response.json({ data: ids.map((id) => ({ id })) });
}

function createRuntime(
	options: {
		credentials?: InMemoryCredentialStore;
		modelsStore?: InMemoryModelsStore;
		env?: Record<string, string | undefined>;
	} = {},
) {
	const credentials = options.credentials ?? new InMemoryCredentialStore();
	const modelsStore = options.modelsStore ?? new InMemoryModelsStore();
	const env = options.env ?? {};
	const authContext: AuthContext = {
		env: async (name) => env[name],
		fileExists: async () => false,
	};
	const models = createModels({ credentials, modelsStore, authContext });
	const provider = openrouterProvider();
	models.setProvider(provider);
	return { models, provider, credentials, modelsStore, env };
}

async function storeCredential(credentials: InMemoryCredentialStore, credential: Credential): Promise<void> {
	await credentials.modify("openrouter", async () => credential);
}

afterEach(() => {
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
	vi.useRealTimers();
});

// Regression coverage for GitHub issue #10353.
describe("OpenRouter workspace model catalog", () => {
	it("keeps complete model metadata while withholding availability before catalog verification", async () => {
		const fetchMock = vi.fn(async () => catalogResponse([chat.id]));
		vi.stubGlobal("fetch", fetchMock);
		const { models, provider } = createRuntime({ env: { OPENROUTER_API_KEY: "workspace-key" } });

		expect(provider.getModels()).toContainEqual(chat);
		expect(provider.getAllModels?.()).toContainEqual(image);
		expect(provider.getAllModels?.()).toContainEqual(classifier);
		expect(models.getModel("openrouter", chat.id)).toEqual(chat);
		expect(await models.getAvailable("openrouter")).toEqual([]);
		expect(await models.getAllAvailable("openrouter")).toEqual([]);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it("does not fetch or advertise models without OpenRouter credentials", async () => {
		const fetchMock = vi.fn(async () => catalogResponse([chat.id]));
		vi.stubGlobal("fetch", fetchMock);
		const { models } = createRuntime();

		expect((await models.refresh({ force: true })).errors.size).toBe(0);
		expect(await models.getAllAvailable("openrouter")).toEqual([]);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it.each(["api_key", "oauth", "env"] as const)(
		"requests the approved catalog with the effective %s credential",
		async (source) => {
			const fetchMock = vi.fn(async (_input: string | URL | Request, _init?: RequestInit) =>
				catalogResponse([chat.id, image.id, classifier.id]),
			);
			vi.stubGlobal("fetch", fetchMock);
			const runtime = createRuntime({ env: { OPENROUTER_API_KEY: "ambient-key" } });
			if (source === "api_key") {
				await storeCredential(runtime.credentials, { type: "api_key", key: "stored-key" });
			} else if (source === "oauth") {
				await storeCredential(runtime.credentials, {
					type: "oauth",
					access: "oauth-key",
					refresh: "",
					expires: Number.MAX_SAFE_INTEGER,
				});
			}

			const result = await runtime.models.refresh({ force: true });

			expect(result.errors.size).toBe(0);
			expect(fetchMock).toHaveBeenCalledTimes(1);
			const [url, init] = fetchMock.mock.calls[0];
			expect(String(url)).toBe("https://openrouter.ai/api/v1/models/user?output_modalities=all");
			expect(new Headers(init?.headers).get("authorization")).toBe(
				`Bearer ${source === "oauth" ? "oauth-key" : source === "api_key" ? "stored-key" : "ambient-key"}`,
			);
			expect((await runtime.models.getAvailable("openrouter")).map((model) => model.id)).toContain(chat.id);
			expect(await runtime.models.getAvailableOfType("image", "openrouter")).toContainEqual(image);
			expect(await runtime.models.getAvailableOfType("classifier", "openrouter")).toContainEqual(classifier);
			const approvedIds = new Set<string>([chat.id, image.id, classifier.id]);
			expect((await runtime.models.getAllAvailable("openrouter")).every((model) => approvedIds.has(model.id))).toBe(
				true,
			);
			const stored = await runtime.modelsStore.read("openrouter");
			expect(stored?.approvedModelIds).toEqual([chat.id, image.id, classifier.id]);
			expect(stored?.credentialScope).toMatch(/^openrouter:[a-f0-9]{64}$/);
			expect(JSON.stringify(stored)).not.toContain("ambient-key");
			expect(JSON.stringify(stored)).not.toContain("stored-key");
			expect(JSON.stringify(stored)).not.toContain("oauth-key");
		},
	);

	it("uses exact approved IDs when filtering custom chat and non-chat models", async () => {
		const privateId = "organization/private-model";
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([privateId])),
		);
		const { models, provider, credentials, modelsStore } = createRuntime();
		const credential: Credential = { type: "api_key", key: "workspace-key" };
		await storeCredential(credentials, credential);
		const approvedChat = { ...chat, id: privateId };
		const unapprovedChat = { ...chat, id: `${privateId}:free` };
		const approvedImage = { ...image, id: privateId };
		const unapprovedClassifier = { ...classifier, id: `${privateId}-other` };

		expect((await models.refresh({ force: true })).errors.size).toBe(0);
		expect(provider.filterModels?.([approvedChat, unapprovedChat], credential)).toEqual([approvedChat]);
		expect(
			provider.filterAllModels?.([approvedChat, unapprovedChat, approvedImage, unapprovedClassifier], credential),
		).toEqual([approvedChat, approvedImage]);
		expect((await modelsStore.read("openrouter"))?.approvedModelIds).toEqual([privateId]);
	});

	it.each(["stored", "env", "config"] as const)("restores a same-credential %s cache offline", async (source) => {
		const credentials = new InMemoryCredentialStore();
		const env: Record<string, string | undefined> =
			source === "stored" ? {} : { OPENROUTER_API_KEY: "workspace-key" };
		if (source === "stored") await storeCredential(credentials, { type: "api_key", key: "workspace-key" });
		if (source === "config") await storeCredential(credentials, { type: "api_key", env: { CUSTOM_CONFIG: "value" } });
		const online = createRuntime({ credentials, env });
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([chat.id])),
		);
		expect((await online.models.refresh({ force: true })).errors.size).toBe(0);
		const cached = await online.modelsStore.read("openrouter");
		const fetchMock = vi.fn(async () => {
			throw new Error("offline cache must not fetch");
		});
		vi.stubGlobal("fetch", fetchMock);
		const offline = createRuntime({ credentials, modelsStore: online.modelsStore, env });

		expect((await offline.models.refresh({ allowNetwork: false })).errors.size).toBe(0);
		expect((await offline.models.getAvailable("openrouter")).map((model) => model.id)).toEqual([chat.id]);
		expect(await offline.modelsStore.read("openrouter")).toEqual(cached);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it("ignores an old unscoped public catalog cache", async () => {
		const runtime = createRuntime({ env: { OPENROUTER_API_KEY: "workspace-key" } });
		await runtime.modelsStore.write("openrouter", { models: [chat], checkedAt: Date.now() });
		const fetchMock = vi.fn(async () => catalogResponse([chat.id]));
		vi.stubGlobal("fetch", fetchMock);

		expect((await runtime.models.refresh({ allowNetwork: false })).errors.size).toBe(0);
		expect(await runtime.models.getAvailable("openrouter")).toEqual([]);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it("does not restore another account's approved cache", async () => {
		const runtime = createRuntime();
		await storeCredential(runtime.credentials, { type: "api_key", key: "first-account" });
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([chat.id])),
		);
		await runtime.models.refresh({ force: true });
		await storeCredential(runtime.credentials, { type: "api_key", key: "second-account" });
		const offline = createRuntime({ credentials: runtime.credentials, modelsStore: runtime.modelsStore });

		expect((await offline.models.refresh({ allowNetwork: false })).errors.size).toBe(0);
		expect(await offline.models.getAllAvailable("openrouter")).toEqual([]);
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);
	});

	it("withholds approvals after an environment-key change and after logout", async () => {
		const runtime = createRuntime({ env: { OPENROUTER_API_KEY: "first-account" } });
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([chat.id])),
		);
		await runtime.models.refresh({ force: true });
		expect((await runtime.models.getAvailable("openrouter")).map((model) => model.id)).toEqual([chat.id]);

		runtime.env.OPENROUTER_API_KEY = "second-account";
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);
		await runtime.models.refresh({ force: true });
		expect((await runtime.models.getAvailable("openrouter")).map((model) => model.id)).toEqual([chat.id]);
		runtime.env.OPENROUTER_API_KEY = undefined;
		await runtime.models.logout("openrouter");
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);
	});

	it.each(["network", "server", "invalid JSON", "invalid catalog"] as const)(
		"retains only the same-account approved cache on a %s failure",
		async (failure) => {
			const runtime = createRuntime({ env: { OPENROUTER_API_KEY: "workspace-key" } });
			vi.stubGlobal(
				"fetch",
				vi.fn(async () => catalogResponse([chat.id])),
			);
			await runtime.models.refresh({ force: true });
			const cached = await runtime.modelsStore.read("openrouter");
			vi.stubGlobal(
				"fetch",
				vi.fn(async () => {
					if (failure === "network") throw new Error("catalog is offline");
					if (failure === "server") return Response.json({ error: "unavailable" }, { status: 503 });
					if (failure === "invalid JSON") return new Response("{");
					return Response.json({ data: null });
				}),
			);

			const result = await runtime.models.refresh({ force: true });

			expect(result.errors.get("openrouter")).toBeInstanceOf(Error);
			expect((await runtime.models.getAvailable("openrouter")).map((model) => model.id)).toEqual([chat.id]);
			expect(await runtime.modelsStore.read("openrouter")).toEqual(cached);
		},
	);

	it.each([401, 403])("clears approvals and persistence when the credential receives HTTP %s", async (status) => {
		const runtime = createRuntime({ env: { OPENROUTER_API_KEY: "workspace-key" } });
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([chat.id])),
		);
		await runtime.models.refresh({ force: true });
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => Response.json({ error: "denied" }, { status })),
		);

		const result = await runtime.models.refresh({ force: true });

		expect(result.errors.get("openrouter")?.message).toContain(String(status));
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);
		expect(await runtime.modelsStore.read("openrouter")).toBeUndefined();
	});

	it("replaces previous approvals with an empty verified catalog", async () => {
		const runtime = createRuntime({ env: { OPENROUTER_API_KEY: "workspace-key" } });
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([chat.id])),
		);
		await runtime.models.refresh({ force: true });
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([])),
		);

		expect((await runtime.models.refresh({ force: true })).errors.size).toBe(0);
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);
		expect((await runtime.modelsStore.read("openrouter"))?.approvedModelIds).toEqual([]);
	});

	it("does not restore rejected approvals when deleting the cache fails", async () => {
		const runtime = createRuntime({ env: { OPENROUTER_API_KEY: "workspace-key" } });
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([chat.id])),
		);
		await runtime.models.refresh({ force: true });
		const cached = await runtime.modelsStore.read("openrouter");
		vi.spyOn(runtime.modelsStore, "delete").mockRejectedValue(new Error("Cache is read-only"));
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => Response.json({ error: "denied" }, { status: 401 })),
		);

		const denied = await runtime.models.refresh({ force: true });

		expect(denied.errors.get("openrouter")).toBeInstanceOf(Error);
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);
		expect(await runtime.modelsStore.read("openrouter")).toEqual(cached);
		expect((await runtime.models.refresh({ allowNetwork: false })).errors.size).toBe(0);
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => {
				throw new Error("Catalog is temporarily offline");
			}),
		);
		expect((await runtime.models.refresh({ force: true })).errors.get("openrouter")).toBeInstanceOf(Error);
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);

		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([otherChat.id])),
		);
		expect((await runtime.models.refresh({ force: true })).errors.size).toBe(0);
		expect((await runtime.models.getAvailable("openrouter")).map((model) => model.id)).toEqual([otherChat.id]);
	});

	it("keeps a verified empty catalog when saving it fails instead of restoring old approvals", async () => {
		const runtime = createRuntime({ env: { OPENROUTER_API_KEY: "workspace-key" } });
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([chat.id])),
		);
		await runtime.models.refresh({ force: true });
		const cached = await runtime.modelsStore.read("openrouter");
		const writeSpy = vi.spyOn(runtime.modelsStore, "write").mockRejectedValue(new Error("Cache is read-only"));
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([])),
		);

		const result = await runtime.models.refresh({ force: true });

		expect(result.errors.get("openrouter")).toBeInstanceOf(Error);
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);
		expect(await runtime.modelsStore.read("openrouter")).toEqual(cached);
		expect((await runtime.models.refresh({ allowNetwork: false })).errors.size).toBe(0);
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);

		writeSpy.mockRestore();
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([otherChat.id])),
		);
		expect((await runtime.models.refresh({ force: true })).errors.size).toBe(0);
		expect((await runtime.models.getAvailable("openrouter")).map((model) => model.id)).toEqual([otherChat.id]);
		expect((await runtime.modelsStore.read("openrouter"))?.approvedModelIds).toEqual([otherChat.id]);
	});

	it("remembers rejected cached approvals across multiple account switches when deletion fails", async () => {
		const runtime = createRuntime({ env: { OPENROUTER_API_KEY: "first-account" } });
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([chat.id])),
		);
		await runtime.models.refresh({ force: true });
		const cached = await runtime.modelsStore.read("openrouter");
		vi.spyOn(runtime.modelsStore, "delete").mockRejectedValue(new Error("Cache is read-only"));
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => Response.json({ error: "denied" }, { status: 401 })),
		);

		expect((await runtime.models.refresh({ force: true })).errors.get("openrouter")).toBeInstanceOf(Error);
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);
		runtime.env.OPENROUTER_API_KEY = "second-account";
		expect((await runtime.models.refresh({ force: true })).errors.get("openrouter")).toBeInstanceOf(Error);
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);
		runtime.env.OPENROUTER_API_KEY = "first-account";
		expect((await runtime.models.refresh({ allowNetwork: false })).errors.size).toBe(0);
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);
		expect(await runtime.modelsStore.read("openrouter")).toEqual(cached);

		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([otherChat.id])),
		);
		expect((await runtime.models.refresh({ force: true })).errors.size).toBe(0);
		runtime.env.OPENROUTER_API_KEY = "second-account";
		await runtime.models.refresh({ allowNetwork: false });
		runtime.env.OPENROUTER_API_KEY = "first-account";
		await runtime.models.refresh({ allowNetwork: false });
		expect((await runtime.models.getAvailable("openrouter")).map((model) => model.id)).toEqual([otherChat.id]);
	});

	it("does not restore stale disk approvals after switching away from a verified catalog that could not be saved", async () => {
		const runtime = createRuntime({ env: { OPENROUTER_API_KEY: "first-account" } });
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([chat.id])),
		);
		await runtime.models.refresh({ force: true });
		const cached = await runtime.modelsStore.read("openrouter");
		vi.spyOn(runtime.modelsStore, "write").mockRejectedValue(new Error("Cache is read-only"));
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([])),
		);

		expect((await runtime.models.refresh({ force: true })).errors.get("openrouter")).toBeInstanceOf(Error);
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);
		runtime.env.OPENROUTER_API_KEY = "second-account";
		expect((await runtime.models.refresh({ allowNetwork: false })).errors.size).toBe(0);
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);
		runtime.env.OPENROUTER_API_KEY = "first-account";
		expect((await runtime.models.refresh({ allowNetwork: false })).errors.size).toBe(0);
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);
		expect(await runtime.modelsStore.read("openrouter")).toEqual(cached);
	});

	it("does not publish a late response after a newer refresh supersedes it", async () => {
		let finishFirst: ((response: Response) => void) | undefined;
		const firstResponse = new Promise<Response>((resolve) => {
			finishFirst = resolve;
		});
		const fetchMock = vi.fn(async () => catalogResponse([otherChat.id]));
		fetchMock.mockImplementationOnce(async () => firstResponse);
		vi.stubGlobal("fetch", fetchMock);
		const runtime = createRuntime({ env: { OPENROUTER_API_KEY: "workspace-key" } });
		const first = runtime.models.refresh({ force: true });
		await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

		const second = await runtime.models.refresh({ force: true });
		await first;
		finishFirst?.(catalogResponse([chat.id]));
		await new Promise((resolve) => setTimeout(resolve, 0));

		expect(second.errors.size).toBe(0);
		expect((await runtime.models.getAvailable("openrouter")).map((model) => model.id)).toEqual([otherChat.id]);
		expect((await runtime.modelsStore.read("openrouter"))?.approvedModelIds).toEqual([otherChat.id]);
	});

	it("returns promptly on cancellation and discards a later successful response", async () => {
		let finishRequest: ((response: Response) => void) | undefined;
		const response = new Promise<Response>((resolve) => {
			finishRequest = resolve;
		});
		const fetchMock = vi.fn(async () => response);
		vi.stubGlobal("fetch", fetchMock);
		const runtime = createRuntime({ env: { OPENROUTER_API_KEY: "workspace-key" } });
		const controller = new AbortController();
		const pending = runtime.models.refresh({ force: true, signal: controller.signal });
		await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

		controller.abort();
		const result = await pending;
		finishRequest?.(catalogResponse([chat.id]));
		await new Promise((resolve) => setTimeout(resolve, 0));

		expect(result.aborted).toBe(true);
		expect(result.errors.size).toBe(0);
		expect(await runtime.models.getAllAvailable("openrouter")).toEqual([]);
		expect(await runtime.modelsStore.read("openrouter")).toBeUndefined();
	});

	it("times out a non-cooperative catalog request while keeping same-account approvals", async () => {
		const runtime = createRuntime({ env: { OPENROUTER_API_KEY: "workspace-key" } });
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => catalogResponse([chat.id])),
		);
		await runtime.models.refresh({ force: true });
		const cached = await runtime.modelsStore.read("openrouter");
		const fetchMock = vi.fn(async () => new Promise<Response>(() => {}));
		vi.stubGlobal("fetch", fetchMock);

		const result = await runtime.models.refresh({ force: true });

		expect(result.aborted).toBe(false);
		expect(result.errors.get("openrouter")?.message).toContain("timed out");
		expect((await runtime.models.getAvailable("openrouter")).map((model) => model.id)).toEqual([chat.id]);
		expect(await runtime.modelsStore.read("openrouter")).toEqual(cached);
	}, 10_000);
});
