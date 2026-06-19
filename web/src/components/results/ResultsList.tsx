import type { SearchResultItem } from "@/api/types";
import { ResultCard } from "@/components/results/ResultCard";

interface ResultsListProps {
  results: SearchResultItem[];
}

export function ResultsList({ results }: ResultsListProps) {
  return (
    <div className="space-y-3">
      {results.map((result, index) => (
        // doc_id can repeat across hybrid fusion edge cases; rank is the
        // true unique key for a single response's ordering.
        <ResultCard key={result.rank} result={result} index={index} />
      ))}
    </div>
  );
}
