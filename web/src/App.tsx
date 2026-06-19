import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BootGate } from '@/components/layout/BootGate';
import { AppShell } from '@/components/layout/AppShell';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      retryOnMount: false,
      refetchOnMount: false,
      refetchOnReconnect: false,
      refetchOnWindowFocus: false,
    },
  },
});

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BootGate>
        <AppShell />
      </BootGate>
    </QueryClientProvider>
  );
}
