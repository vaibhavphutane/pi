import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { InMemoryCredentialStore, InMemoryModelsStore } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CredentialSynchronizationError, ModelRuntime } from "../src/core/model-runtime.ts";
import { allowNetwork } from "./test-network-env.ts";

const approvedId = "anthropic/claude-sonnet-4";
const blockedId = "anthropic/claude-opus-4.1";
const catalogUrl = "https://openrouter.ai/api/v1/models/user?output_modalities=all";
const tempDirs: string[] = [];

function catalog(ids: readonly string[]): Response {
	return new Response(
		JSON.stringify({
			data: ids.map((id) => ({
				id,
				name: id,
				architecture: { input_modalities: ["text"], output_modalities: ["text"] },
				context_length: 200_000,
				top_provider: { context_length: 200_000, max_completion_tokens: 16_000 },
				supported_parameters: ["tools"],
				pricing: { prompt: "0.000003", completion: "0.000015" },
			})),
		}),
		{ headers: { "content-type": "application/json" } },
	);
}

function modelsPath(provider: object): string {
	const dir = mkdtempSync(join(tmpdir(), "pi-openrouter-catalog-"));
	tempDirs.push(dir);
	const path = join(dir, "models.json");
	writeFileSync(path, JSON.stringify({ providers: { openrouter: provider } }));
	return path;
}

function availableIds(runtime: ModelRuntime): string[] {
	return runtime
		.getAvailableSnapshot()
		.filter((model) => model.provider === "openrouter")
		.map((model) => model.id);
}

afterEach(() => {
	vi.restoreAllMocks();
	for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

// Regression coverage for GitHub issue #10353.
describe("ModelRuntime OpenRouter approved catalog", () => {
	it("uses the authenticated catalog instead of the public pi.dev overlay", async () => {
		allowNetwork();
		const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => catalog([approvedId]));
		const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null });
		await runtime.setRuntimeApiKey("openrouter", "workspace-key");

		// Bundled metadata supports synchronous lookup; approval controls availability.
		expect(runtime.getModel("openrouter", blockedId)).toBeDefined();
		expect(availableIds(runtime)).toEqual([]);
		expect(fetchSpy).not.toHaveBeenCalled();

		const result = await runtime.refresh({ allowNetwork: true, providers: ["openrouter"] });
		expect([...result.errors]).toEqual([]);
		expect(availableIds(runtime)).toEqual([approvedId]);
		expect((await runtime.getAvailable("openrouter")).map((model) => model.id)).toEqual([approvedId]);
		expect(fetchSpy).toHaveBeenCalledOnce();
		expect(String(fetchSpy.mock.calls[0]?.[0])).toBe(catalogUrl);
		expect(new Headers(fetchSpy.mock.calls[0]?.[1]?.headers).get("authorization")).toBe("Bearer workspace-key");
	});

	it("applies approval to custom models and preserves it during provider recomposition", async () => {
		allowNetwork();
		vi.spyOn(globalThis, "fetch").mockImplementation(async () => catalog([approvedId, "custom/approved"]));
		const runtime = await ModelRuntime.create({
			credentials: new InMemoryCredentialStore(),
			modelsStore: new InMemoryModelsStore(),
			modelsPath: modelsPath({
				apiKey: "workspace-key",
				models: [{ id: "custom/approved" }, { id: "custom/blocked" }],
				modelOverrides: { [approvedId]: { name: "Approved Sonnet" } },
			}),
		});
		expect(availableIds(runtime)).toEqual([]);
		expect(runtime.getProvider("openrouter")?.credentialScopedModels).toBe(true);
		await runtime.refresh({ allowNetwork: true, providers: ["openrouter"] });
		expect(availableIds(runtime)).toEqual([approvedId, "custom/approved"]);
		expect(runtime.getAvailableSnapshot().find((model) => model.id === approvedId)?.name).toBe("Approved Sonnet");

		// Registration publishes a provisional snapshot before its async availability pass.
		runtime.registerProvider("openrouter", { headers: { "X-Title": "pi-test" } });
		expect(availableIds(runtime)).toEqual([approvedId, "custom/approved"]);
		await runtime.refresh({ allowNetwork: false, providers: ["openrouter"] });
		expect(availableIds(runtime)).toEqual([approvedId, "custom/approved"]);

		const approved = runtime.getModel("openrouter", approvedId);
		if (!approved) throw new Error("Missing approved model metadata");
		runtime.registerProvider("openrouter", {
			models: [approved, { ...approved, id: "custom/blocked", name: "Blocked" }],
		});
		expect(availableIds(runtime)).toEqual([approvedId]);
		await runtime.refresh({ allowNetwork: false, providers: ["openrouter"] });
		expect(availableIds(runtime)).toEqual([approvedId]);

		runtime.registerProvider("openrouter", { apiKey: "another-workspace-key" });
		expect(availableIds(runtime)).toEqual([]);
		await runtime.refresh({ allowNetwork: false, providers: ["openrouter"] });
		expect(availableIds(runtime)).toEqual([]);
	});

	it("retains only the same credential's approved models after a refresh failure", async () => {
		allowNetwork();
		const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => catalog([approvedId]));
		const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null });
		await runtime.setRuntimeApiKey("openrouter", "workspace-one");
		await runtime.refresh({ allowNetwork: true, providers: ["openrouter"] });

		fetchSpy.mockRejectedValue(new Error("Catalog temporarily unavailable"));
		const failed = await runtime.refresh({ allowNetwork: true, force: true, providers: ["openrouter"] });
		expect(failed.errors.has("openrouter")).toBe(true);
		expect(availableIds(runtime)).toEqual([approvedId]);

		await runtime.setRuntimeApiKey("openrouter", "workspace-two");
		expect(availableIds(runtime)).toEqual([]);
		const changed = await runtime.refresh({ allowNetwork: true, providers: ["openrouter"] });
		expect(changed.errors.has("openrouter")).toBe(true);
		expect(availableIds(runtime)).toEqual([]);
	});

	it("clears prior workspace approvals when credential synchronization cannot read its cache", async () => {
		allowNetwork();
		const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => catalog([approvedId]));
		const store = new InMemoryModelsStore();
		const runtime = await ModelRuntime.create({
			credentials: new InMemoryCredentialStore(),
			modelsStore: store,
			modelsPath: null,
		});
		await runtime.setRuntimeApiKey("openrouter", "workspace-one");
		await runtime.refresh({ allowNetwork: true, providers: ["openrouter"] });
		expect(availableIds(runtime)).toEqual([approvedId]);
		fetchSpy.mockClear();
		vi.spyOn(store, "read").mockRejectedValue(new Error("Catalog cache unavailable"));

		await expect(runtime.setRuntimeApiKey("openrouter", "workspace-two")).rejects.toBeInstanceOf(
			CredentialSynchronizationError,
		);
		expect(availableIds(runtime)).toEqual([]);
		expect(fetchSpy).not.toHaveBeenCalled();
	});

	it("restores approval offline only for matching credentials and clears it on logout", async () => {
		allowNetwork();
		const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => catalog([approvedId]));
		const credentials = new InMemoryCredentialStore();
		await credentials.modify("openrouter", async () => ({ type: "api_key", key: "workspace-one" }));
		const store = new InMemoryModelsStore();
		const options = { credentials, modelsStore: store, modelsPath: null };
		const runtime = await ModelRuntime.create(options);
		await runtime.refresh({ allowNetwork: true, providers: ["openrouter"] });
		const stored = await store.read("openrouter");
		expect(stored?.credentialScope).toEqual(expect.any(String));
		expect(JSON.stringify(stored)).not.toContain("workspace-one");

		fetchSpy.mockClear();
		const restored = await ModelRuntime.create(options);
		expect(availableIds(restored)).toEqual([approvedId]);
		await restored.logout("openrouter");
		expect(availableIds(restored)).toEqual([]);
		expect(fetchSpy).not.toHaveBeenCalled();

		await credentials.modify("openrouter", async () => ({ type: "api_key", key: "workspace-two" }));
		const changed = await ModelRuntime.create(options);
		expect(availableIds(changed)).toEqual([]);
		expect(fetchSpy).not.toHaveBeenCalled();
	});

	it("does not treat an old public catalog cache as workspace approval", async () => {
		const credentials = new InMemoryCredentialStore();
		await credentials.modify("openrouter", async () => ({ type: "api_key", key: "workspace-key" }));
		const store = new InMemoryModelsStore();
		const runtime = await ModelRuntime.create({ credentials, modelsPath: null, modelsStore: store });
		const model = runtime.getModel("openrouter", approvedId);
		if (!model) throw new Error("Missing bundled model metadata");
		await store.write("openrouter", { models: [model], checkedAt: Date.now(), lastModified: Date.now() });
		const fetchSpy = vi.spyOn(globalThis, "fetch");

		await runtime.refresh({ allowNetwork: false, providers: ["openrouter"] });
		expect(availableIds(runtime)).toEqual([]);
		expect(fetchSpy).not.toHaveBeenCalled();
	});

	it("resolves configured environment keys before offline restoration and availability filtering", async () => {
		allowNetwork();
		vi.stubEnv("PI_TEST_OPENROUTER_KEY", "workspace-one");
		vi.spyOn(globalThis, "fetch").mockImplementation(async () => catalog([approvedId]));
		const credentials = new InMemoryCredentialStore();
		const store = new InMemoryModelsStore();
		const options = {
			credentials,
			modelsStore: store,
			modelsPath: modelsPath({ apiKey: "$PI_TEST_OPENROUTER_KEY" }),
		};
		const runtime = await ModelRuntime.create(options);
		await runtime.refresh({ allowNetwork: true, providers: ["openrouter"] });
		expect(availableIds(runtime)).toEqual([approvedId]);
		const restored = await ModelRuntime.create(options);
		expect(availableIds(restored)).toEqual([approvedId]);

		vi.stubEnv("PI_TEST_OPENROUTER_KEY", "workspace-two");
		expect(await runtime.getAvailable("openrouter")).toEqual([]);
		await runtime.refresh({ allowNetwork: false, providers: ["openrouter"] });
		expect(availableIds(runtime)).toEqual([]);
	});
});
