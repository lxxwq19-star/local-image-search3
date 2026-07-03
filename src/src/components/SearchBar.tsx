import { useState, useRef } from 'react';

interface SearchBarProps {
  onSearch: (query: string, mode: 'text' | 'image') => void;
  loading: boolean;
  mode: 'text' | 'image';
  onModeChange: (mode: 'text' | 'image') => void;
  onSelectFile: () => void;
}

function SearchBar({ onSearch, loading, mode, onModeChange, onSelectFile }: SearchBarProps) {
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    onSearch(query.trim(), 'text');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && query.trim()) {
      onSearch(query.trim(), 'text');
    }
  };

  return (
    <div className="search-module">
      {/* Tab 切换 */}
      <div className="tab-bar">
        <button
          className={`tab-item ${mode === 'text' ? 'active' : ''}`}
          onClick={() => onModeChange('text')}
        >
          语义搜图
        </button>
        <button
          className={`tab-item ${mode === 'image' ? 'active' : ''}`}
          onClick={() => onModeChange('image')}
        >
          以图搜图
        </button>
      </div>

      {/* 语义搜图面板 */}
      {mode === 'text' && (
        <div className="search-panel" id="textPanel">
          <div className="search-input-wrapper">
            <svg className="search-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8"/>
              <path d="M21 21l-4.35-4.35"/>
            </svg>
            <input
              ref={inputRef}
              type="text"
              placeholder="输入描述，搜索本地图片（例：颜色，形状，品类）"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
            />
            {query && (
              <button className="clear-input-btn" onClick={() => setQuery('')}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18"/>
                  <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </button>
            )}
          </div>
          <button
            className="primary-btn"
            onClick={handleSubmit}
            disabled={loading || !query.trim()}
          >
            {loading ? (
              <>
                <span className="loading-spinner" style={{ marginRight: 8 }}></span>
                搜索中...
              </>
            ) : '开始搜索'}
          </button>
        </div>
      )}

      {/* 以图搜图面板 */}
      {mode === 'image' && (
        <div className="search-panel" id="imagePanel">
          <button
            className="upload-card"
            onClick={onSelectFile}
            disabled={loading}
          >
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <path d="M21 15l-5-5L5 21"/>
            </svg>
            <p>点击选择本地图片，匹配相似图片</p>
          </button>
        </div>
      )}
    </div>
  );
}

export default SearchBar;
