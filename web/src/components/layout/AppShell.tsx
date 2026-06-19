import { useUiStore } from "@/stores/uiStore";
import { Header } from "@/components/layout/Header";
import { SearchView } from "@/components/views/SearchView";
import { EvaluationView } from "@/components/evaluation/EvaluationView";

export function AppShell() {
  const activeView = useUiStore((s) => s.activeView);

  return (
    <div className="min-h-dvh bg-bg">
      <Header />
      <main>{activeView === "search" ? <SearchView /> : <EvaluationView />}</main>
    </div>
  );
}
