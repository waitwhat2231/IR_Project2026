import { useRef, useState } from 'react';
import type { KeyboardEvent, SyntheticEvent } from 'react';
import { Search, SlidersHorizontal, X } from 'lucide-react';
import clsx from 'clsx';
import { useSuggestions } from '@/hooks/useSuggestions';
import { useUiStore } from '@/stores/uiStore';
import { IconButton } from '@/components/ui/IconButton';
import { Spinner } from '@/components/ui/Spinner';

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (query: string) => void;
  disabled?: boolean;
  isSearching?: boolean;
  variant?: 'hero' | 'compact';
  onOpenSettings: () => void;
  autoFocus?: boolean;
}

export function SearchBar({
  value,
  onChange,
  onSubmit,
  disabled = false,
  isSearching = false,
  variant = 'compact',
  onOpenSettings,
  autoFocus = false,
}: SearchBarProps) {
  const [focused, setFocused] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const settingsPanelOpen = useUiStore((s) => s.settingsPanelOpen);

  const suggestionsQuery = useSuggestions(value);
  const suggestions = suggestionsQuery.data?.suggestions ?? [];
  const showSuggestions =
    focused && value.trim().length >= 2 && suggestions.length > 0;

  const commitSearch = (query: string): void => {
    const trimmed = query.trim();
    if (!trimmed) return;
    setFocused(false);
    setHighlightIndex(-1);
    inputRef.current?.blur();
    onSubmit(trimmed);
  };

  const handleSubmit = (event: SyntheticEvent<HTMLFormElement>): void => {
    event.preventDefault();
    commitSearch(value);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (!showSuggestions) return;

    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setHighlightIndex((index) => Math.min(index + 1, suggestions.length - 1));
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      setHighlightIndex((index) => Math.max(index - 1, -1));
    } else if (event.key === 'Escape') {
      setFocused(false);
    } else if (event.key === 'Enter' && highlightIndex >= 0) {
      event.preventDefault();
      const picked = suggestions[highlightIndex];
      if (picked) {
        onChange(picked);
        commitSearch(picked);
      }
    }
  };

  return (
    <div className="relative w-full">
      <form
        onSubmit={handleSubmit}
        className={clsx(
          'flex items-center gap-2 rounded-control border bg-surface px-2 transition-colors duration-200',
          focused ? 'border-accent/60' : 'border-border',
          disabled && 'pointer-events-none opacity-50',
          variant === 'hero' ? 'h-14 px-3' : 'h-12',
        )}
      >
        <Search
          size={variant === 'hero' ? 19 : 17}
          className="shrink-0 text-text-muted"
        />

        <input
          ref={inputRef}
          type="text"
          inputMode="search"
          autoFocus={autoFocus}
          value={value}
          disabled={disabled}
          placeholder="Search the corpus — e.g. “should teachers get tenure?”"
          onChange={(event) => {
            onChange(event.target.value);
            setHighlightIndex(-1);
          }}
          onFocus={() => {
            setFocused(true);
          }}
          onBlur={() => {
            // Defer so a suggestion click can register before the list unmounts.
            setTimeout(() => {
              setFocused(false);
            }, 120);
          }}
          onKeyDown={handleKeyDown}
          className={clsx(
            'min-w-0 flex-1 bg-transparent text-text placeholder:text-text-muted focus:outline-none',
            variant === 'hero' ? 'text-base' : 'text-sm',
          )}
        />

        {isSearching && <Spinner size={16} className="shrink-0 text-accent" />}

        {value.length > 0 && !isSearching && (
          <button
            type="button"
            aria-label="Clear query"
            onClick={() => {
              onChange('');
              inputRef.current?.focus();
            }}
            className="shrink-0 cursor-pointer rounded-full p-1 text-text-muted transition-colors hover:bg-surface-hover hover:text-text"
          >
            <X size={14} />
          </button>
        )}

        <div className="h-6 w-px shrink-0 bg-border" />

        <IconButton
          type="button"
          aria-label="Search and model settings"
          active={settingsPanelOpen}
          onClick={onOpenSettings}
          className="h-8 w-8"
        >
          <SlidersHorizontal size={16} />
        </IconButton>
      </form>

      {showSuggestions && (
        <ul
          role="listbox"
          className="animate-fade-in absolute left-0 right-0 top-[calc(100%+8px)] z-30 overflow-hidden rounded-control border border-border-strong bg-surface-elevated shadow-2xl"
        >
          {suggestions.map((suggestion, index) => (
            <li key={suggestion}>
              <button
                type="button"
                role="option"
                aria-selected={index === highlightIndex}
                onMouseDown={(event) => {
                  event.preventDefault();
                }}
                onClick={() => {
                  onChange(suggestion);
                  commitSearch(suggestion);
                }}
                className={clsx(
                  'flex w-full cursor-pointer items-center gap-2.5 px-3.5 py-2.5 text-left text-sm text-text-secondary transition-colors',
                  index === highlightIndex
                    ? 'bg-accent-subtle text-text'
                    : 'hover:bg-surface-hover hover:text-text',
                )}
              >
                <Search size={13} className="shrink-0 text-text-muted" />
                <span className="truncate">{suggestion}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
