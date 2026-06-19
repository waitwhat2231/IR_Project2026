import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { copyToClipboard } from "@/utils/clipboard";

interface CopyButtonProps {
  value: string;
  label: string;
}

const CONFIRMATION_MS = 1500;

export function CopyButton({ value, label }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleClick = async (): Promise<void> => {
    const success = await copyToClipboard(value);
    if (success) {
      setCopied(true);
      setTimeout(() => {
        setCopied(false);
      }, CONFIRMATION_MS);
    }
  };

  return (
    <button
      type="button"
      onClick={() => {
        void handleClick();
      }}
      aria-label={copied ? `${label} copied` : `Copy ${label}`}
      className="inline-flex h-6 w-6 shrink-0 cursor-pointer items-center justify-center rounded text-text-muted transition-colors duration-150 hover:bg-surface-hover hover:text-accent-warm"
    >
      {copied ? (
        <Check size={13} className="text-success" />
      ) : (
        <Copy size={13} />
      )}
    </button>
  );
}
