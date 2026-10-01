import { InMemoryCredentialStore } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { listModels } from "../src/cli/list-models.ts";
import { ModelRuntime } from "../src/core/model-runtime.ts";
import { allowNetwork } from "./test-network-env.ts";

async function createRuntime() {
	const credentials = new InMemoryCredentialStore();
	await credentials.modify("openrouter", async () => ({ type: "api_key", key: "workspace-key" }));
	const runtime = await ModelRuntime.create({ credentials, modelsPath: null, allowModelNetwork: false });
	const [approved, blocked] = runtime.getModels("openrouter");
	if (!approved || !blocked) throw new Error("Expected the bundled OpenRouter model metadata");
	return { runtime, credentials, approved, blocked };
}

describe("--list-models OpenRouter workspace catalog", () => {
	afterEach(() => {
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it("fetches the authenticated catalog before printing only approved models", async () => {
		allowNetwork();
		const { runtime, approved, blocked } = await createRuntime();
		const fetch = vi.fn<typeof globalThis.fetch>(async () => Response.json({ data: [{ id: approved.id }] }));
		vi.stubGlobal("fetch", fetch);
		const output = vi.spyOn(console, "log").mockImplementation(() => {});
		const refresh = vi.spyOn(runtime, "refresh");

		await listModels(runtime);

		expect(refresh).toHaveBeenCalledWith({
			providers: ["openrouter"],
			allowNetwork: true,
			force: true,
			signal: undefined,
		});
		expect(fetch).toHaveBeenCalledOnce();
		const [input, init] = fetch.mock.calls[0];
		const request = new Request(input, init);
		expect(request.url).toBe("https://openrouter.ai/api/v1/models/user?output_modalities=all");
		expect(request.headers.get("authorization")).toBe("Bearer workspace-key");
		const rendered = output.mock.calls.flat().join("\n");
		expect(rendered).toContain(approved.id);
		expect(rendered).not.toContain(blocked.id);
	});

	it("retains only previously approved models when a catalog refresh fails", async () => {
		allowNetwork();
		const { runtime, approved, blocked } = await createRuntime();
		const fetch = vi.fn<typeof globalThis.fetch>(async () => Response.json({ data: [{ id: approved.id }] }));
		vi.stubGlobal("fetch", fetch);
		const output = vi.spyOn(console, "log").mockImplementation(() => {});
		const warnings = vi.spyOn(console, "error").mockImplementation(() => {});
		await listModels(runtime);
		output.mockClear();
		fetch.mockImplementation(async () => new Response("unavailable", { status: 503 }));

		await listModels(runtime);

		const rendered = output.mock.calls.flat().join("\n");
		expect(rendered).toContain(approved.id);
		expect(rendered).not.toContain(blocked.id);
		expect(warnings.mock.calls.flat().join("\n")).toContain("could not refresh OpenRouter models");
	});

	it("keeps other providers available when the first OpenRouter catalog request fails", async () => {
		allowNetwork();
		const { runtime, approved, blocked } = await createRuntime();
		runtime.registerProvider("local-test", {
			baseUrl: "https://local.test/v1",
			api: "openai-completions",
			apiKey: "test-key",
			models: [
				{
					id: "local-model",
					name: "Local Model",
					reasoning: false,
					input: ["text"],
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					contextWindow: 4096,
					maxTokens: 2048,
				},
			],
		});
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => new Response("unavailable", { status: 503 })),
		);
		const output = vi.spyOn(console, "log").mockImplementation(() => {});
		const warnings = vi.spyOn(console, "error").mockImplementation(() => {});

		await listModels(runtime);

		const rendered = output.mock.calls.flat().join("\n");
		expect(rendered).toContain("local-model");
		expect(rendered).not.toContain(approved.id);
		expect(rendered).not.toContain(blocked.id);
		expect(warnings.mock.calls.flat().join("\n")).toContain("could not refresh OpenRouter models");
	});

	it("does not make network requests in offline mode", async () => {
		const { runtime, approved, blocked } = await createRuntime();
		const fetch = vi.fn<typeof globalThis.fetch>(async () => Response.json({ data: [{ id: approved.id }] }));
		vi.stubGlobal("fetch", fetch);
		const output = vi.spyOn(console, "log").mockImplementation(() => {});
		const refresh = vi.spyOn(runtime, "refresh");

		await listModels(runtime);

		expect(refresh).toHaveBeenCalledWith({
			providers: ["openrouter"],
			allowNetwork: false,
			force: true,
			signal: undefined,
		});
		expect(fetch).not.toHaveBeenCalled();
		const rendered = output.mock.calls.flat().join("\n");
		expect(rendered).not.toContain(approved.id);
		expect(rendered).not.toContain(blocked.id);
	});

	it("removes revoked models when a refreshed workspace catalog is empty", async () => {
		allowNetwork();
		const { runtime, approved } = await createRuntime();
		const fetch = vi.fn<typeof globalThis.fetch>(async () => Response.json({ data: [{ id: approved.id }] }));
		vi.stubGlobal("fetch", fetch);
		const output = vi.spyOn(console, "log").mockImplementation(() => {});
		await listModels(runtime);
		output.mockClear();
		fetch.mockImplementation(async () => Response.json({ data: [] }));

		await listModels(runtime);

		expect(output.mock.calls.flat().join("\n")).not.toContain(approved.id);
	});

	it("does not reuse an earlier workspace catalog after the credential changes", async () => {
		allowNetwork();
		const { runtime, credentials, approved } = await createRuntime();
		const fetch = vi.fn<typeof globalThis.fetch>(async () => Response.json({ data: [{ id: approved.id }] }));
		vi.stubGlobal("fetch", fetch);
		const output = vi.spyOn(console, "log").mockImplementation(() => {});
		vi.spyOn(console, "error").mockImplementation(() => {});
		await listModels(runtime);
		output.mockClear();
		await credentials.modify("openrouter", async () => ({ type: "api_key", key: "another-workspace-key" }));
		fetch.mockImplementation(async () => new Response("unavailable", { status: 503 }));

		await listModels(runtime);

		expect(output.mock.calls.flat().join("\n")).not.toContain(approved.id);
	});

	it("honors caller cancellation before starting the catalog refresh", async () => {
		const { runtime } = await createRuntime();
		const refresh = vi.spyOn(runtime, "refresh");
		const controller = new AbortController();
		controller.abort();

		await expect(listModels(runtime, undefined, controller.signal)).rejects.toMatchObject({ name: "AbortError" });
		expect(refresh).not.toHaveBeenCalled();
	});
});
