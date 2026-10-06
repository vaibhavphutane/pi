import { afterEach, describe, expect, it, vi } from "vitest";
import type { AgentSession } from "../../src/core/agent-session.ts";
import { createAgentSession } from "../../src/core/sdk.ts";
import { createTestResourceLoader } from "../utilities.ts";
import { createHarness, type Harness } from "./harness.ts";

// Regression coverage for GitHub issue #10353.
describe("SDK restoration from credential-scoped catalogs", () => {
	let harness: Harness | undefined;
	let restoredSession: AgentSession | undefined;

	afterEach(() => {
		restoredSession?.dispose();
		restoredSession = undefined;
		harness?.cleanup();
		harness = undefined;
		vi.restoreAllMocks();
	});

	async function restoreSession(options: {
		savedId: "approved" | "revoked";
		availableIds?: string[];
		credentialScoped?: boolean;
		explicitId?: string;
	}) {
		harness = await createHarness({ models: [{ id: "approved" }, { id: "revoked" }] });
		const runtime = harness.session.modelRuntime;
		const provider = runtime.getProvider(harness.models[0].provider);
		if (!provider) throw new Error("Expected faux provider");
		runtime.registerNativeProvider({ ...provider, credentialScopedModels: options.credentialScoped ?? true });
		const availableIds = options.availableIds ?? ["approved"];
		vi.spyOn(runtime, "getAvailableSnapshot").mockReturnValue(
			harness.models.filter((model) => availableIds.includes(model.id)),
		);
		harness.settingsManager.setDefaultModelAndProvider(provider.id, "approved");
		harness.sessionManager.appendModelChange(provider.id, options.savedId);
		harness.sessionManager.appendMessage({ role: "user", content: "Saved conversation", timestamp: Date.now() });
		const result = await createAgentSession({
			cwd: harness.tempDir,
			agentDir: harness.tempDir,
			modelRuntime: runtime,
			model: options.explicitId ? harness.getModel(options.explicitId) : undefined,
			sessionManager: harness.sessionManager,
			settingsManager: harness.settingsManager,
			resourceLoader: createTestResourceLoader(),
			noTools: "all",
		});
		restoredSession = result.session;
		return result;
	}

	it("falls back from a revoked session model to an approved saved default", async () => {
		const result = await restoreSession({ savedId: "revoked" });
		expect(result.session.model?.id).toBe("approved");
		expect(result.modelFallbackMessage).toContain("Could not restore model");
		expect(result.modelFallbackMessage).toContain("Using");
	});

	it("restores an approved recorded model", async () => {
		const result = await restoreSession({ savedId: "approved" });
		expect(result.session.model?.id).toBe("approved");
		expect(result.modelFallbackMessage).toBeUndefined();
	});

	it("uses the unknown-model placeholder when no catalog approvals are available", async () => {
		const result = await restoreSession({ savedId: "revoked", availableIds: [] });
		expect(result.session.model?.provider).toBe("unknown");
		expect(result.modelFallbackMessage).toBeDefined();
	});

	it("preserves explicit SDK model selection", async () => {
		const result = await restoreSession({ savedId: "approved", explicitId: "revoked" });
		expect(result.session.model?.id).toBe("revoked");
		expect(result.modelFallbackMessage).toBeUndefined();
	});

	it("preserves restoration for providers without a credential-scoped catalog", async () => {
		const result = await restoreSession({ savedId: "revoked", availableIds: [], credentialScoped: false });
		expect(result.session.model?.id).toBe("revoked");
		expect(result.modelFallbackMessage).toBeUndefined();
	});
});
