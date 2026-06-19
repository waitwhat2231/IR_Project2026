import { create } from 'zustand';

export type AppView = 'search' | 'evaluation' | 'clusters';

interface UiState {
  activeView: AppView;
  settingsPanelOpen: boolean;
  setActiveView: (view: AppView) => void;
  toggleSettingsPanel: () => void;
  closeSettingsPanel: () => void;
}

export const useUiStore = create<UiState>()((set) => ({
  activeView: 'search',
  settingsPanelOpen: false,

  setActiveView: (activeView) => {
    set({ activeView });
  },
  toggleSettingsPanel: () => {
    set((state) => ({ settingsPanelOpen: !state.settingsPanelOpen }));
  },
  closeSettingsPanel: () => {
    set({ settingsPanelOpen: false });
  },
}));
