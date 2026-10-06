import type { Api, Model } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { InteractiveMode } from "../../src/modes/interactive/interactive-mode.ts";
import { createHarness, type Harness } from "./harness.ts";

const findExactModelMatch = Reflect.get(InteractiveMode.prototype, "findExactModelMatch") as (
	this: object,
	searchTerm: string,
) => Promise<Model<Api> | undefined>;

function scopeCatalogToCredentials(harness: Harness, initialModels: readonly Model<Api>[]) {
	const runtime = harness.session.modelRuntime;
	const provider = runtime.getProvider(harness.models[0].provider);
	if (!provider) throw new Error("Expected faux provider");
	runtime.registerNativeProvider({ ...provider, credentialScopedModels: true });
	let availableModels = initialModels;
	vi.spyOn(runtime, "getAvailableSnapshot").mockImplementation(() => availableModels);
	return (models: readonly Model<Api>[]) => {
		availableModels = models;
	};
}

// Regression coverage for GitHub issue #10353.
describe("credential-scoped session models", () => {
	let harness: Harness | undefined;

	afterEach(() => {
		harness?.cleanup();
		harness = undefined;
		vi.restoreAllMocks();
	});

	it("hides revoked saved scopes and restores their thinking settings when approved again", async () => {
		harness = await createHarness({ models: [{ id: "approved" }, { id: "revoked" }] });
		const [approved, revoked] = harness.models;
		const updateAvailable = scopeCatalogToCredentials(harness, [approved]);
		const savedScope = [{ model: approved }, { model: revoked, thinkingLevel: "high" as const }];
		const savedIds = savedScope.map(({ model }) => `${model.provider}/${model.id}`);
		harness.settingsManager.setEnabledModels(savedIds);
		harness.session.setScopedModels(savedScope);
		const extensionContext = harness.session.extensionRunner.createContext();

		expect(harness.session.scopedModels).toEqual([{ model: approved }]);
		expect(extensionContext.scopedModels).toEqual([{ model: approved }]);
		expect(harness.settingsManager.getEnabledModels()).toEqual(savedIds);

		updateAvailable([approved, revoked]);
		expect(harness.session.scopedModels).toEqual(savedScope);
		expect(extensionContext.scopedModels).toEqual(savedScope);

		updateAvailable([]);
		expect(harness.session.scopedModels).toEqual([]);
		expect(extensionContext.scopedModels).toEqual([]);
		expect(harness.settingsManager.getEnabledModels()).toEqual(savedIds);
	});

	it("does not resolve a revoked model from cached interactive scopes", async () => {
		harness = await createHarness({ models: [{ id: "approved" }, { id: "revoked" }] });
		const [approved, revoked] = harness.models;
		const updateAvailable = scopeCatalogToCredentials(harness, [approved, revoked]);
		harness.session.setScopedModels([{ model: approved }, { model: revoked }]);
		const refresh = vi.spyOn(harness.session.modelRuntime, "refresh");
		const context = { session: harness.session, showStatus: vi.fn(), showWarning: vi.fn() };
		const revokedReference = `${revoked.provider}/${revoked.id}`;

		expect(await findExactModelMatch.call(context, revokedReference)).toEqual(revoked);
		updateAvailable([approved]);
		expect(await findExactModelMatch.call(context, revokedReference)).toBeUndefined();
		expect(await findExactModelMatch.call(context, approved.id)).toEqual(approved);
		expect(refresh).not.toHaveBeenCalled();
	});

	it("preserves scopes for providers without a credential-scoped catalog", async () => {
		harness = await createHarness({ models: [{ id: "saved" }] });
		vi.spyOn(harness.session.modelRuntime, "getAvailableSnapshot").mockReturnValue([]);
		const savedScope = [{ model: harness.models[0] }];
		harness.session.setScopedModels(savedScope);
		expect(harness.session.scopedModels).toEqual(savedScope);
	});
});
