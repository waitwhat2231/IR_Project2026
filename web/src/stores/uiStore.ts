import { create } from "zustand";

export type AppView = "search" | "evaluation";

interface UiState {
  activeView: AppView;
  settingsPanelOpen: boolean;
  rawJsonOpen: boolean;
  setActiveView: (view: AppView) => void;
  toggleSettingsPanel: () => void;
  closeSettingsPanel: () => void;
  toggleRawJson: () => void;
}

export const useUiStore = create<UiState>()((set) => ({
  activeView: "search",
  settingsPanelOpen: false,
  rawJsonOpen: false,

  setActiveView: (activeView) => {
    set({ activeView });
  },
  toggleSettingsPanel: () => {
    set((state) => ({ settingsPanelOpen: !state.settingsPanelOpen }));
  },
  closeSettingsPanel: () => {
    set({ settingsPanelOpen: false });
  },
  toggleRawJson: () => {
    set((state) => ({ rawJsonOpen: !state.rawJsonOpen }));
  },
}));
