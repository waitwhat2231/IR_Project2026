"""
Hand-built, geometry-only line icons (no emoji, no external icon fonts).
Every shape is composed of plain SVG primitives so nothing can render
as a broken/empty glyph.
"""

from __future__ import annotations

_ICONS: dict[str, str] = {
    "search": '<circle cx="11" cy="11" r="6"></circle>'
    '<line x1="20" y1="20" x2="15.4" y2="15.4"></line>',
    "target": '<circle cx="12" cy="12" r="8"></circle>'
    '<circle cx="12" cy="12" r="4"></circle>'
    '<circle cx="12" cy="12" r="0.6" fill="currentColor" stroke="none"></circle>',
    "doc": '<rect x="5" y="3" width="14" height="18" rx="2"></rect>'
    '<line x1="8" y1="8" x2="16" y2="8"></line>'
    '<line x1="8" y1="12" x2="16" y2="12"></line>'
    '<line x1="8" y1="16" x2="13" y2="16"></line>',
    "sliders": '<line x1="4" y1="6" x2="20" y2="6"></line>'
    '<circle cx="9" cy="6" r="2" fill="currentColor"></circle>'
    '<line x1="4" y1="12" x2="20" y2="12"></line>'
    '<circle cx="16" cy="12" r="2" fill="currentColor"></circle>'
    '<line x1="4" y1="18" x2="20" y2="18"></line>'
    '<circle cx="11" cy="18" r="2" fill="currentColor"></circle>',
    "layers": '<ellipse cx="12" cy="6" rx="7" ry="2.6"></ellipse>'
    '<ellipse cx="12" cy="12" rx="7" ry="2.6"></ellipse>'
    '<ellipse cx="12" cy="18" rx="7" ry="2.6"></ellipse>',
    "list": '<line x1="9" y1="6" x2="20" y2="6"></line>'
    '<line x1="9" y1="12" x2="20" y2="12"></line>'
    '<line x1="9" y1="18" x2="20" y2="18"></line>'
    '<circle cx="4.5" cy="6" r="1" fill="currentColor"></circle>'
    '<circle cx="4.5" cy="12" r="1" fill="currentColor"></circle>'
    '<circle cx="4.5" cy="18" r="1" fill="currentColor"></circle>',
    "clock": '<circle cx="12" cy="12" r="9"></circle>'
    '<line x1="12" y1="7.5" x2="12" y2="12"></line>'
    '<line x1="12" y1="12" x2="15.2" y2="14"></line>',
    "bars": '<rect x="4" y="13" width="3.2" height="7" rx="1" fill="currentColor" stroke="none"></rect>'
    '<rect x="10.4" y="8" width="3.2" height="12" rx="1" fill="currentColor" stroke="none"></rect>'
    '<rect x="16.8" y="3" width="3.2" height="17" rx="1" fill="currentColor" stroke="none"></rect>',
    "venn": '<circle cx="9" cy="12" r="6.2"></circle>'
    '<circle cx="15" cy="12" r="6.2"></circle>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"></rect>'
    '<line x1="9" y1="2" x2="9" y2="6"></line><line x1="15" y1="2" x2="15" y2="6"></line>'
    '<line x1="9" y1="18" x2="9" y2="22"></line><line x1="15" y1="18" x2="15" y2="22"></line>'
    '<line x1="2" y1="9" x2="6" y2="9"></line><line x1="2" y1="15" x2="6" y2="15"></line>'
    '<line x1="18" y1="9" x2="22" y2="9"></line><line x1="18" y1="15" x2="22" y2="15"></line>',
    "zap": '<polygon points="13,2 3,14 12,14 11,22 21,10 12,10" fill="currentColor" stroke="none"></polygon>',
    "sparkle": '<polygon points="12,2 14.12,9.88 22,12 14.12,14.12 12,22 9.88,14.12 2,12 9.88,9.88" '
    'fill="currentColor" stroke="none"></polygon>',
    "check": '<circle cx="12" cy="12" r="9"></circle>'
    '<polyline points="8,12 11,15 16,9"></polyline>',
    "server": '<rect x="3" y="4" width="18" height="6" rx="1.5"></rect>'
    '<rect x="3" y="14" width="18" height="6" rx="1.5"></rect>'
    '<circle cx="7" cy="7" r="0.8" fill="currentColor" stroke="none"></circle>'
    '<circle cx="7" cy="17" r="0.8" fill="currentColor" stroke="none"></circle>',
    "alert": '<circle cx="12" cy="12" r="9"></circle>'
    '<line x1="12" y1="8" x2="12" y2="13"></line>'
    '<circle cx="12" cy="16.2" r="0.6" fill="currentColor" stroke="none"></circle>',
}


def _icon(name: str, size: int = 15, cls: str = "") -> str:
    """Render a small inline SVG icon. Falls back to a plain circle if unknown."""
    inner = _ICONS.get(name, '<circle cx="12" cy="12" r="9"></circle>')
    return (
        f'<svg class="ir-icon {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="1.75" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'xmlns="http://www.w3.org/2000/svg">{inner}</svg>'
    )


def _section_label(text: str, icon_name: str) -> str:
    return f'<div class="ir-sec">{_icon(icon_name, 13)}<span>{text}</span></div>'


def _eyebrow(text: str, icon_name: str) -> str:
    return f'<div class="ir-eyebrow">{_icon(icon_name, 13)}<span>{text}</span></div>'
