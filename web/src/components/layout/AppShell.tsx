import { useUiStore } from "@/stores/uiStore";
import { Header } from "@/components/layout/Header";
import { SearchView } from "@/components/views/SearchView";
import { EvaluationView } from "@/components/evaluation/EvaluationView";
import { ClustersView } from "@/components/clusters/ClustersView";

export function AppShell() {
  const activeView = useUiStore((s) => s.activeView);

  return (
    <div className="min-h-dvh bg-bg">
      <Header />
      <main>
        {activeView === "search" && <SearchView />}
        {activeView === "evaluation" && <EvaluationView />}
        {activeView === "clusters" && <ClustersView />}
      </main>
    </div>
  );
}
