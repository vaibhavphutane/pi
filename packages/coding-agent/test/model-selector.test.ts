import { setKeybindings, type TUI } from "@earendil-works/pi-tui";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { KeybindingsManager } from "../src/core/keybindings.ts";
import { ModelSelectorComponent } from "../src/modes/interactive/components/model-selector.ts";
import { initTheme } from "../src/modes/interactive/theme/theme.ts";
import { stripAnsi } from "../src/utils/ansi.ts";
import { createHarness, type Harness } from "./suite/harness.ts";

function createFakeTui(): TUI {
	return { requestRender: () => {} } as unknown as TUI;
}

describe("model selector", () => {
	let harness: Harness | undefined;

	beforeAll(() => {
		initTheme("dark");
	});

	beforeEach(() => {
		setKeybindings(new KeybindingsManager());
	});

	afterEach(() => {
		harness?.cleanup();
		harness = undefined;
		vi.restoreAllMocks();
	});

	it("keeps the current model marked while browsing", async () => {
		harness = await createHarness({
			models: [
				{ id: "current-model", name: "Current Model", reasoning: true },
				{ id: "browsed-model", name: "Browsed Model", reasoning: true },
			],
		});
		const currentModel = harness.getModel("current-model")!;
		const selector = new ModelSelectorComponent(
			createFakeTui(),
			currentModel,
			harness.session.modelRuntime,
			[],
			() => {},
			() => {},
		);

		const getModelRow = (id: string): string | undefined =>
			stripAnsi(selector.render(120).join("\n"))
				.split("\n")
				.find((line) => line.includes(`${id} [`))
				?.trimEnd();

		expect(getModelRow("current-model")).toBe(`→ ✓ current-model [${currentModel.provider}]`);
		selector.handleInput("\x1b[B");
		expect(getModelRow("current-model")).toBe(`  ✓ current-model [${currentModel.provider}]`);
		expect(getModelRow("browsed-model")).toBe(`→   browsed-model [${currentModel.provider}]`);
		selector.dispose();
	});

	it("uses the configured save binding", async () => {
		setKeybindings(new KeybindingsManager({ "app.models.save": "ctrl+r" }));
		harness = await createHarness();
		const currentModel = harness.getModel()!;
		const saveDefault = vi.fn();
		const selector = new ModelSelectorComponent(
			createFakeTui(),
			currentModel,
			harness.session.modelRuntime,
			[],
			() => {},
			() => {},
			undefined,
			saveDefault,
		);

		expect(stripAnsi(selector.render(120).join("\n"))).toContain("Ctrl+R to set as default");
		selector.handleInput("\x13");
		expect(saveDefault).not.toHaveBeenCalled();
		selector.handleInput("\x12");
		expect(saveDefault).toHaveBeenCalledWith(currentModel);
	});

	it("lists every catalog that failed to refresh", async () => {
		harness = await createHarness();
		vi.spyOn(harness.session.modelRuntime, "refresh").mockResolvedValue({
			aborted: false,
			errors: new Map([
				["openai", new Error("unavailable")],
				["anthropic", new Error("unavailable")],
			]),
		});

		const selector = new ModelSelectorComponent(
			createFakeTui(),
			harness.getModel(),
			harness.session.modelRuntime,
			[],
			() => {},
			() => {},
		);

		await vi.waitFor(() => {
			const rendered = stripAnsi(selector.render(120).join("\n"));
			expect(rendered).toContain("Could not refresh 2 model catalogs (openai, anthropic); showing cached models.");
		});
		selector.dispose();
	});

	it("excludes unapproved OpenRouter models from a saved scope", async () => {
		harness = await createHarness();
		const local = harness.getModel();
		const unapproved = { ...local, provider: "openrouter", id: "unapproved/model" };
		vi.spyOn(harness.session.modelRuntime, "getAvailableSnapshot").mockReturnValue([local]);
		vi.spyOn(harness.session.modelRuntime, "getModel").mockImplementation((provider, id) =>
			provider === "openrouter" && id === unapproved.id ? unapproved : local,
		);
		vi.spyOn(harness.session.modelRuntime, "refresh").mockResolvedValue({ aborted: false, errors: new Map() });
		const selector = new ModelSelectorComponent(
			createFakeTui(),
			local,
			harness.session.modelRuntime,
			[{ model: unapproved }, { model: local }],
			() => {},
			() => {},
		);

		expect(stripAnsi(selector.render(120).join("\n"))).not.toContain("unapproved/model");
		expect(stripAnsi(selector.render(120).join("\n"))).toContain(`${local.id} [${local.provider}]`);
		await vi.waitFor(() => expect(stripAnsi(selector.render(120).join("\n"))).toContain("Model catalogs refreshed."));
		expect(stripAnsi(selector.render(120).join("\n"))).not.toContain("unapproved/model");
		selector.dispose();
	});

	it("removes revoked scoped OpenRouter entries after refreshing and allows switching to all models", async () => {
		harness = await createHarness();
		const local = harness.getModel();
		const revoked = { ...local, provider: "openrouter", id: "revoked/model" };
		let available = [revoked, local];
		vi.spyOn(harness.session.modelRuntime, "getAvailableSnapshot").mockImplementation(() => available);
		vi.spyOn(harness.session.modelRuntime, "getModel").mockReturnValue(revoked);
		vi.spyOn(harness.session.modelRuntime, "refresh").mockImplementation(async () => {
			available = [local];
			return { aborted: false, errors: new Map() };
		});
		const selected = vi.fn();
		const selector = new ModelSelectorComponent(
			createFakeTui(),
			revoked,
			harness.session.modelRuntime,
			[{ model: revoked }],
			selected,
			() => {},
		);

		await vi.waitFor(() => expect(stripAnsi(selector.render(120).join("\n"))).toContain("Model catalogs refreshed."));
		expect(stripAnsi(selector.render(120).join("\n"))).not.toContain("revoked/model");
		selector.handleInput("\r");
		expect(selected).not.toHaveBeenCalled();
		selector.handleInput("\t");
		expect(stripAnsi(selector.render(120).join("\n"))).toContain(`${local.id} [${local.provider}]`);
		selector.handleInput("\r");
		expect(selected).toHaveBeenCalledWith(local);
	});

	it("reloads filtered availability when refreshing rejects after a credential change", async () => {
		harness = await createHarness();
		const local = harness.getModel();
		const previous = { ...local, provider: "openrouter", id: "previous-workspace/model" };
		let available = [previous, local];
		vi.spyOn(harness.session.modelRuntime, "getAvailableSnapshot").mockImplementation(() => available);
		vi.spyOn(harness.session.modelRuntime, "getModel").mockReturnValue(previous);
		vi.spyOn(harness.session.modelRuntime, "refresh").mockImplementation(async () => {
			available = [local];
			throw new Error("workspace changed");
		});
		const selector = new ModelSelectorComponent(
			createFakeTui(),
			previous,
			harness.session.modelRuntime,
			[{ model: previous }],
			() => {},
			() => {},
		);

		await vi.waitFor(() => {
			const rendered = stripAnsi(selector.render(120).join("\n"));
			expect(rendered).toContain("Could not refresh model catalogs: workspace changed");
			expect(rendered).not.toContain("previous-workspace/model");
		});
		selector.dispose();
	});
});
