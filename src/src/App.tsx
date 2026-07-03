import { useState, useEffect } from 'react';
import { invoke, convertFileSrc } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import SearchBar from './components/SearchBar';
import './styles.css';

interface ImageResult {
  id: number;
  path: string;
  similarity?: number;
  width?: number;
  height?: number;
}

interface IndexProgress {
  counted: number;
  indexed: number;
  errors?: number;
  status: string;
  current_file?: string;
  eta_seconds?: number;  // Estimated time remaining (seconds)
}

interface IndexStatus {
  indexed_count: number;
  index_size: number;
  status: string;
}

interface IndexPath {
  id: number;
  path: string;
  name: string;
  enabled: boolean;
  indexed_count: number;
}

function App() {
  const [currentPage, setCurrentPage] = useState<'home' | 'results'>('home');
  const [searchResults, setSearchResults] = useState<ImageResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [indexing, setIndexing] = useState(false);
  const [indexProgress, setIndexProgress] = useState<IndexProgress | null>(null);
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [indexPaths, setIndexPaths] = useState<IndexPath[]>([]);
  const [mode, setMode] = useState<'text' | 'image'>('text');
  const [searchQuery, setSearchQuery] = useState('');

  // 缩略图缩放
  const [thumbSize, setThumbSize] = useState(160);

  // 设置弹窗
  const [showSettings, setShowSettings] = useState(false);

  // 历史记录
  const [searchHistory, setSearchHistory] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem('searchHistory') || '[]');
    } catch {
      return [];
    }
  });

  // 子文件夹展开状态
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set());
  const [subfolderLists, setSubfolderLists] = useState<Record<string, any[]>>({});

  // 二次确认弹窗状态
  const [confirmDialog, setConfirmDialog] = useState<{
    show: boolean;
    title: string;
    message: string;
    onConfirm: () => void;
  }>({ show: false, title: '', message: '', onConfirm: () => {} });

  // 自定义确认弹窗
  const showConfirm = (title: string, message: string, onConfirm: () => void) => {
    setConfirmDialog({ show: true, title, message, onConfirm });
  };

  const closeConfirm = () => {
    setConfirmDialog(prev => ({ ...prev, show: false }));
  };

  // 展开/折叠子文件夹列表
  const handleToggleExpand = async (path: string) => {
    const next = new Set(expandedPaths);
    if (next.has(path)) {
      // 已展开 → 折叠
      next.delete(path);
      setExpandedPaths(next);
    } else {
      // 未展开 → 先展开，再加载子文件夹
      next.add(path);
      setExpandedPaths(next);
      if (!subfolderLists[path]) {
        try {
          const result = await invoke<{ subfolders: any[] }>('get_subfolders', { rootPath: path });
          const list = result.subfolders || [];
          if (list.length > 0) {
            setSubfolderLists(prev => ({ ...prev, [path]: list }));
          } else {
            // 没有子文件夹，自动折叠
            const next2 = new Set(expandedPaths);
            next2.delete(path);
            setExpandedPaths(next2);
          }
        } catch (err) {
          console.warn('[App] Failed to load subfolders:', err);
          // 加载失败也折叠
          const next2 = new Set(expandedPaths);
          next2.delete(path);
          setExpandedPaths(next2);
        }
      }
    }
  };

  // 切换子文件夹启用状态
  const handleToggleSubfolder = async (subfolderPath: string) => {
    try {
      await invoke<boolean>('toggle_subfolder', { subfolderPath: subfolderPath });
      // 刷新对应的子文件夹列表
      const rootPath = Object.keys(subfolderLists).find(key =>
        subfolderLists[key]?.some((sf: any) => sf.subfolder_path === subfolderPath)
      );
      if (rootPath) {
        const result = await invoke<{ subfolders: any[] }>('get_subfolders', { rootPath: rootPath });
        setSubfolderLists(prev => ({ ...prev, [rootPath]: result.subfolders || [] }));
      }
    } catch (err) {
      console.error('[App] Failed to toggle subfolder:', err);
    }
  };

  // 首次使用引导
  const [showFirstTime, setShowFirstTime] = useState(() => {
    return !localStorage.getItem('hasSeenFirstTime');
  });

  useEffect(() => {
    loadIndexStatus();
    loadPaths();
    setupProgressListener();
    setupDragDropListener();
  }, []);

  // 加载路径列表
  const loadPaths = async () => {
    try {
      const result = await invoke<{ paths: IndexPath[] }>('get_paths');
      setIndexPaths(result.paths || []);
    } catch (err) {
      console.warn('[App] loadPaths failed:', err);
    }
  };

  const handleSearch = async (query: string, searchMode: 'text' | 'image') => {
    // 索引中搜索需要二次确认
    if (indexing) {
      showConfirm(
        '索引进行中',
        '索引正在进行中，搜索可能导致响应变慢。确定要继续搜索吗？',
        async () => {
          closeConfirm();
          await doSearch(query, searchMode);
        }
      );
      return;
    }
    await doSearch(query, searchMode);
  };

  // 带超时的 invoke 包装（默认 3 分钟）
  async function invokeWithTimeout<T>(cmd: string, args: any, timeoutMs: number = 180000): Promise<T> {
    return Promise.race([
      invoke<T>(cmd, args),
      new Promise<T>((_, reject) =>
        setTimeout(
          () => reject(new Error(`"${cmd}" 超时（${Math.round(timeoutMs/1000)}秒），请检查图片是否过大或格式不支持`)),
          timeoutMs
        )
      ),
    ]);
  }

  const doSearch = async (query: string, searchMode: 'text' | 'image') => {
    setLoading(true);
    setError(null);
    try {
      let results;
      if (searchMode === 'text') {
        results = await invoke<{ images: ImageResult[] }>('search_by_text', {
          query,
          topK: 9999,
        });
      } else {
        results = await invokeWithTimeout<{ images: ImageResult[] }>('search_by_image', {
          imagePath: query,
          topK: 9999,
        }, 180000);
      }
      setSearchResults(results.images || []);
      setSearchQuery(searchMode === 'text' ? query : '以图搜图');
      if (searchMode === 'text') {
        addToHistory(query);
      }
      setCurrentPage('results');
    } catch (err) {
      console.error('搜索失败:', err);
      setError(`搜索失败: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectFile = async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp', 'heic', 'avif'] }],
        title: '选择查询图片',
      });
      if (selected) {
        handleSearch(selected as string, 'image');
      }
    } catch (err) {
      console.error('选择文件失败:', err);
      setError(`选择文件失败: ${err}`);
    }
  };

  // 添加索引路径
  const handleAddPath = async () => {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: '选择要索引的文件夹',
      });
      if (selected) {
        // 提取文件夹名称作为 display name
        const folderName = (selected as string).split(/[/\\]/).pop() || selected as string;
        await invoke('add_path', { path: selected, name: folderName });
        await loadPaths();
        // 自动开始索引导弹
        setIndexing(true);
        setIndexProgress({ counted: 0, indexed: 0, status: 'scanning' });
        await invoke<string>('index_images', { directory: selected, force_reencode: true });
      }
    } catch (err) {
      console.error('添加路径失败:', err);
      setError(`添加路径失败: ${err}`);
      setIndexing(false);
    }
  };

  // 删除索引路径
  const handleDeletePath = async (path: string) => {
    showConfirm(
      '删除索引路径',
      `确定要删除 "${path}" 及其索引数据吗？此操作不可恢复。`,
      async () => {
        closeConfirm();
        try {
          await invoke('delete_path', { path });
          await loadPaths();
          await loadIndexStatus();
        } catch (err) {
          console.error('删除路径失败:', err);
          setError(`删除路径失败: ${err}`);
        }
      }
    );
  };

  // 切换路径启用状态
  const handleTogglePath = async (path: string) => {
    try {
      await invoke<boolean>('toggle_path', { path });
      await loadPaths();
      await loadIndexStatus();
    } catch (err) {
      console.error('切换路径状态失败:', err);
      setError(`切换路径状态失败: ${err}`);
    }
  };

  // 重建所有索引
  const handleRebuildAll = async () => {
    showConfirm(
      '重建所有索引',
      '确定要重建所有已启用路径的索引吗？这将重新编码所有图片的向量，可能需要较长时间。',
      async () => {
        closeConfirm();
        try {
          setIndexing(true);
          setIndexProgress({ counted: 0, indexed: 0, status: 'scanning' });
          await invoke<string>('rebuild_all_index');
        } catch (err) {
          console.error('重建索引失败:', err);
          setError(`重建索引失败: ${err}`);
          setIndexing(false);
        }
      }
    );
  };

  const setupProgressListener = async () => {
    try {
      await listen<IndexProgress>('index-progress', (event) => {
        setIndexProgress(event.payload);
        if (event.payload.status === 'completed') {
          setIndexing(false);
          loadIndexStatus();
          loadPaths();
        }
      });
    } catch (err) {
      console.log('无法设置进度监听:', err);
    }
  };

  const loadIndexStatus = async () => {
    try {
      const status = await invoke<IndexStatus>('get_index_status');
      setIndexStatus(status);
    } catch (err) {
      console.log('无法加载索引状态:', err);
    }
  };

  const setupDragDropListener = async () => {
    try {
      await listen<{ paths: string[]; position: { x: number; y: number } }>('tauri://drag-drop', (event) => {
        const paths = event.payload?.paths;
        if (paths && paths.length > 0) {
          setMode('image');
          handleSearch(paths[0], 'image');
        }
      });
    } catch (err) {
      console.warn('[App] Could not set up drag-drop listener:', err);
    }
  };

  const addToHistory = (query: string) => {
    const newHistory = [query, ...searchHistory.filter(h => h !== query)];
    setSearchHistory(newHistory);
    localStorage.setItem('searchHistory', JSON.stringify(newHistory));
  };

  const removeHistoryItem = (query: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const newHistory = searchHistory.filter(h => h !== query);
    setSearchHistory(newHistory);
    localStorage.setItem('searchHistory', JSON.stringify(newHistory));
  };

  const clearHistory = () => {
    setSearchHistory([]);
    localStorage.removeItem('searchHistory');
  };

  const searchFromHistory = (query: string) => {
    handleSearch(query, 'text');
  };

  const goHome = () => {
    setCurrentPage('home');
    setError(null);
  };

  const openFile = async (path: string) => {
    try {
      await invoke('open_file', { path });
    } catch (err) {
      console.error('打开文件失败:', err);
    }
  };

  // Ctrl + 滚轮缩放缩略图
  const handleResultsWheel = (e: React.WheelEvent) => {
    if (e.ctrlKey) {
      e.preventDefault();
      setThumbSize(prev => Math.max(80, Math.min(480, prev - e.deltaY * 0.5)));
    }
  };

  // 关闭设置弹窗（索引中退出确认）
  const closeSettings = () => {
    if (indexing) {
      showConfirm(
        '索引进行中',
        '索引正在进行中，关闭设置窗口不会中断索引。确定要关闭吗？',
        () => {
          closeConfirm();
          setShowSettings(false);
        }
      );
      return;
    }
    setShowSettings(false);
  };

  // ========== 辅助函数 ==========

  // 格式化 ETA 秒数为可读字符串
  const formatEta = (seconds: number): string => {
    if (seconds < 60) return `${seconds} 秒`;
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    if (m < 60) return s > 0 ? `${m} 分 ${s} 秒` : `${m} 分钟`;
    const h = Math.floor(m / 60);
    const rm = m % 60;
    return rm > 0 ? `${h} 小时 ${rm} 分` : `${h} 小时`;
  };

  // ========== 首页 ==========
  const renderIndexingBanner = () => {
    if (!indexing || !indexProgress) return null;
    const eta = indexProgress.eta_seconds;
    return (
      <div style={{
        background: 'linear-gradient(90deg, #2196F3, #1976D2)',
        color: '#fff',
        padding: '8px 20px',
        fontSize: 13,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 12,
        boxShadow: '0 2px 8px rgba(33,150,243,0.3)',
      }}>
        <span>⏳ 索引进行中：{indexProgress.indexed?.toLocaleString() || 0} / {indexProgress.counted.toLocaleString()}</span>
        {eta !== undefined && eta > 0 && (
          <span style={{ opacity: 0.9 }}>· 剩余约 {formatEta(eta)}</span>
        )}
      </div>
    );
  };

  const renderHome = () => (
    <div id="page-home" className={`page ${currentPage === 'home' ? 'active' : ''}`}>
      {/* 索引进行中提示 */}
      {renderIndexingBanner()}
      {/* 顶部导航栏 */}
      <header className="app-header">
        <div className="header-left"></div>
        <h1 className="app-title">本地搜</h1>
        <button className="icon-btn header-right" onClick={() => setShowSettings(true)} title="设置">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3"/>
            <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/>
          </svg>
        </button>
      </header>

      <div className="home-content">
        {/* 搜索功能区 */}
        <SearchBar
          onSearch={handleSearch}
          loading={loading}
          mode={mode}
          onModeChange={setMode}
          onSelectFile={handleSelectFile}
        />

        {/* 索引状态提示 */}
        {indexPaths.length === 0 && !indexing && (
          <div style={{
            background: '#fff',
            borderRadius: 16,
            padding: '20px 24px',
            boxShadow: '0 2px 12px rgba(0,0,0,0.04)',
            marginBottom: 24,
            textAlign: 'center',
          }}>
            <p style={{ color: '#666', fontSize: 14, marginBottom: 12 }}>
              还没有索引任何文件夹
            </p>
            <button className="secondary-btn" onClick={handleAddPath}>
              📁 选择文件夹开始索引
            </button>
          </div>
        )}

        {/* 历史记录区 */}
        <div className="history-section">
          <div className="history-header">
            <span className="history-title">搜索历史</span>
            {searchHistory.length > 0 && (
              <button className="text-btn-small" onClick={clearHistory}>清空全部</button>
            )}
          </div>
          {searchHistory.length > 0 ? (
            <div className="history-tags">
              {searchHistory.map((q, i) => (
                <div key={i} className="history-tag-wrapper">
                  <button className="history-tag" onClick={() => searchFromHistory(q)}>
                    {q}
                  </button>
                  <button
                    className="history-tag-delete"
                    onClick={(e) => removeHistoryItem(q, e)}
                    title="删除此记录"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="history-empty">暂无搜索记录</div>
          )}
        </div>

        {/* 高级设置入口 */}
        <div style={{textAlign: 'center', marginTop: 32, marginBottom: 16}}>
          <button className="text-btn-small" onClick={() => setShowSettings(true)} style={{color: '#888', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4, margin: '0 auto'}}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="3"/>
              <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/>
            </svg>
            高级设置（模型选择、索引管理）
          </button>
        </div>
      </div>
    </div>
  );

  // ========== 结果页 ==========
  const renderResults = () => (
    <div id="page-results" className={`page ${currentPage === 'results' ? 'active' : ''}`}>
      {/* 顶部返回栏 */}
      <header className="result-header">
        <button className="icon-btn" onClick={goHome}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5M12 19l-7-7 7-7"/>
          </svg>
        </button>
        <h2 className="result-title">
          {searchQuery ? `"${searchQuery}" 的搜索结果` : '搜索结果'}
        </h2>
        <button className="icon-btn" onClick={() => setShowSettings(true)}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>
          </svg>
        </button>
      </header>

      <div className="results-content">
        {/* 结果统计 */}
        <div className="results-stats">
          找到 {searchResults.length} 个结果
          {indexStatus && (
            <span style={{ marginLeft: 12, color: '#4CAF50' }}>
              已索引 {indexStatus.indexed_count.toLocaleString()} 张图片
            </span>
          )}
          <span style={{ marginLeft: 12, color: '#999', fontSize: 12 }}>
            按住 Ctrl + 滚轮缩放 · 点击直接用系统查看器打开
          </span>
        </div>

        {/* 空结果 */}
        {searchResults.length === 0 ? (
          <div className="empty-result">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
              <circle cx="11" cy="11" r="8"/>
              <line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            <p>未找到匹配图片，请更换描述重试</p>
          </div>
        ) : (
          <div
            className="image-grid"
            style={{ '--thumb-size': `${thumbSize}px` } as React.CSSProperties}
            onWheel={handleResultsWheel}
          >
            {searchResults.map((item) => (
              <div
                key={item.id}
                className="image-grid-item"
                onClick={() => openFile(item.path)}
                title={item.path}
              >
                <img
                  src={convertFileSrc(item.path)}
                  alt=""
                  loading="lazy"
                  style={{ width: thumbSize, height: thumbSize }}
                />
                {item.similarity !== undefined && (
                  <span className="similarity-badge">
                    {(item.similarity * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  // ========== 设置弹窗 ==========
  const renderSettings = () => (
    <div
      className={`modal-overlay ${showSettings ? '' : 'hidden'}`}
      onClick={(e) => { if (e.target === e.currentTarget) closeSettings(); }}
    >
      <div className="modal-content settings-modal">
        <div className="modal-header">
          <h3>索引路径管理</h3>
          <button className="icon-btn" onClick={closeSettings}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"/>
              <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>
        <div className="modal-body">
          {/* 路径列表 */}
          <div className="paths-section">
            <div className="paths-header">
              <span>已索引文件夹</span>
              <button className="text-btn" onClick={handleAddPath}>+ 添加</button>
            </div>
            <div className="paths-list">
              {indexPaths.length === 0 ? (
                <div className="path-item">
                  <span style={{ color: 'var(--text-tertiary)', fontSize: 13 }}>暂无索引路径，点击上方「添加」开始</span>
                </div>
              ) : (
                indexPaths.map((p) => (
                  <div key={p.id}>
                    {/* 路径 item */}
                    <div className={`path-item ${!p.enabled ? 'path-item-disabled' : ''}`}>
                      {/* 展开按钮：只要索引过就显示 */}
                      {p.indexed_count > 0 && (
                        <button
                          className="path-expand"
                          onClick={(e) => { e.stopPropagation(); handleToggleExpand(p.path); }}
                          title="展开子文件夹"
                        >
                          {expandedPaths.has(p.path) ? '▼' : '▶'}
                        </button>
                      )}

                      {/* 启用/禁用开关 */}
                      <button
                        className={`path-toggle ${p.enabled ? 'on' : ''}`}
                        onClick={() => handleTogglePath(p.path)}
                        title={p.enabled ? '点击禁用' : '点击启用'}
                      >
                        <span className="toggle-dot"></span>
                      </button>

                      {/* 路径信息 */}
                      <div className="path-info">
                        <div className="path-name" title={p.path}>
                          {p.name || p.path.split(/[/\\]/).pop()}
                        </div>
                        <div className="path-category" title={p.path}>
                          {p.path}
                        </div>
                        <div className="path-count">
                          {p.indexed_count > 0 ? `${p.indexed_count.toLocaleString()} 张图片` : '未索引'}
                        </div>
                      </div>

                      {/* 操作按钮 */}
                      <div className="path-actions">
                        <button
                          className="path-action-btn danger"
                          onClick={() => handleDeletePath(p.path)}
                          title="删除"
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polyline points="3 6 5 6 21 6"/>
                            <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a1 1 0 011-1h4a1 1 0 011 1v2"/>
                          </svg>
                        </button>
                      </div>
                    </div>

                    {/* 子文件夹列表 */}
                    {expandedPaths.has(p.path) && (
                      <div className="subfolder-list">
                        {subfolderLists[p.path] && subfolderLists[p.path].length > 0 ? (
                          subfolderLists[p.path].map((sf: any) => (
                            <div key={sf.subfolder_path} className={`subfolder-item ${!sf.enabled ? 'subfolder-item-disabled' : ''}`}>
                              <button
                                className={`subfolder-toggle ${sf.enabled ? 'on' : ''}`}
                                onClick={() => handleToggleSubfolder(sf.subfolder_path)}
                                title={sf.enabled ? '点击禁用' : '点击启用'}
                              >
                                <span className="toggle-dot"></span>
                              </button>
                              <span className="subfolder-name" title={sf.subfolder_path}>
                                {sf.subfolder_path.split(/[/\\]/).pop()}
                              </span>
                              <span className="subfolder-count">
                                {sf.indexed_count > 0 ? `${sf.indexed_count} 张` : '未索引'}
                              </span>
                            </div>
                          ))
                        ) : (
                          <div style={{ padding: '8px 16px 8px 36px', fontSize: 12, color: '#999' }}>
                            未扫描到子文件夹
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>

          {/* 模型状态 */}
          <div className="model-section" style={{ marginTop: 20, padding: '12px 0', borderTop: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: 13, color: 'var(--text-tertiary)', marginBottom: 8, fontWeight: 600 }}>
              🧠 向量模型
            </div>
            <div style={{ fontSize: 12, color: '#888', lineHeight: 1.8 }}>
              <div>以图搜图：<span style={{ color: '#4CAF50', fontWeight: 600 }}>SigLIP2-Large-256</span>（1024维，GPU加速）</div>
              <div>语义搜图：<span style={{ color: '#4CAF50', fontWeight: 600 }}>CLIP ViT-L/14</span>（768维，GPU加速）</div>
            </div>
          </div>

          {/* 索引进度条 */}
          {indexProgress && indexing && (
            <div style={{ marginTop: 16 }}>
              <div style={{
                display: 'flex', justifyContent: 'space-between',
                fontSize: 12, color: '#666', marginBottom: 6
              }}>
                <span>
                  {indexProgress.status === 'scanning' ? '🔍 扫描文件中...' :
                   indexProgress.status === 'encoding' ? '🧠 CLIP 向量编码中...' :
                   indexProgress.status === 'completed' ? '✅ 索引完成' :
                   indexProgress.status === 'started' ? '🚀 开始索引...' :
                   `⏳ ${indexProgress.status}`}
                </span>
                <span>
                  {indexProgress.counted > 0
                    ? `${Math.min(100, Math.round(indexProgress.indexed / indexProgress.counted * 100))}%`
                    : ''}
                </span>
              </div>
              {/* 进度条背景 */}
              <div style={{
                height: 8, borderRadius: 4, background: '#e0e0e0', overflow: 'hidden'
              }}>
                {/* 进度条填充 */}
                <div style={{
                  height: '100%',
                  width: indexProgress.counted > 0
                    ? `${Math.min(100, Math.round(indexProgress.indexed / indexProgress.counted * 100))}%`
                    : '0%',
                  background: indexProgress.status === 'completed'
                    ? '#4CAF50'
                    : 'linear-gradient(90deg, #2196F3, #4CAF50)',
                  borderRadius: 4,
                  transition: 'width 0.3s ease',
                }} />
              </div>
              {/* 详情文字 */}
              <div style={{ fontSize: 12, color: '#999', marginTop: 6, display: 'flex', flexDirection: 'column', gap: 4 }}>
                {indexProgress.status === 'scanning'
                  ? `已发现 ${indexProgress.counted.toLocaleString()} 个图片文件`
                  : indexProgress.status === 'encoding'
                  ? (
                    <span>
                      {`向量编码中 ${indexProgress.indexed?.toLocaleString() || 0} / ${indexProgress.counted.toLocaleString()}`}
                      {indexProgress.eta_seconds !== undefined && indexProgress.eta_seconds > 0 && (
                        <span style={{ marginLeft: 12, color: '#2196F3' }}>
                          {` · 剩余约 ${formatEta(indexProgress.eta_seconds)}`}
                        </span>
                      )}
                    </span>
                  )
                  : `已索引 ${indexProgress.indexed?.toLocaleString() || 0} / ${indexProgress.counted.toLocaleString()} 张图片${indexProgress.errors ? `，${indexProgress.errors} 个错误` : ''}`}
              </div>
            </div>
          )}

          {/* 底部操作 */}
          <div className="settings-actions" style={{ marginTop: 20 }}>
            <button
              className="secondary-btn"
              onClick={handleRebuildAll}
              disabled={indexing || indexPaths.filter(p => p.enabled).length === 0}
            >
              {indexing ? '索引中...' : '重建所有索引'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  // ========== 首次使用引导 ==========
  const renderFirstTime = () => (
    <div className={`modal-overlay ${showFirstTime ? '' : 'hidden'}`}>
      <div className="modal-content first-time">
        <div className="modal-body first-time">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/>
          </svg>
          <h3>选择图片文件夹</h3>
          <p>请选择要搜索的本地图片文件夹，我们将为您建立索引</p>
          <button
            className="primary-btn"
            onClick={() => {
              setShowFirstTime(false);
              localStorage.setItem('hasSeenFirstTime', 'true');
              handleAddPath();
            }}
          >
            选择文件夹
          </button>
        </div>
      </div>
    </div>
  );

  // ========== 自定义确认弹窗 ==========
  const renderConfirmDialog = () => (
    <div className={`modal-overlay ${confirmDialog.show ? '' : 'hidden'}`} style={{ zIndex: 4000 }}>
      <div className="modal-content confirm-modal" style={{ maxWidth: 420, padding: '24px 28px' }}>
        <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12, color: '#1a1a1a' }}>
          {confirmDialog.title}
        </h3>
        <p style={{ fontSize: 14, color: '#666', lineHeight: 1.6, marginBottom: 24 }}>
          {confirmDialog.message}
        </p>
        <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
          <button
            className="secondary-btn"
            onClick={closeConfirm}
            style={{ minWidth: 80 }}
          >
            取消
          </button>
          <button
            className="primary-btn"
            onClick={confirmDialog.onConfirm}
            style={{ minWidth: 80, background: '#ff4d4f', borderColor: '#ff4d4f' }}
          >
            确定
          </button>
        </div>
      </div>
    </div>
  );

  // ========== 主渲染 ==========
  return (
    <div className="app">
      {/* 错误提示 */}
      {error && (
        <div style={{
          position: 'fixed',
          top: 16,
          left: '50%',
          transform: 'translateX(-50%)',
          padding: '12px 24px',
          background: '#ff4d4f',
          color: '#fff',
          borderRadius: 8,
          zIndex: 3000,
          fontSize: 14,
          boxShadow: '0 4px 16px rgba(0,0,0,0.2)',
        }}>
          {error}
          <button
            style={{ marginLeft: 16, background: 'none', border: 'none', color: '#fff', cursor: 'pointer' }}
            onClick={() => setError(null)}
          >
            ✕
          </button>
        </div>
      )}

      {renderHome()}
      {renderResults()}
      {renderSettings()}
      {renderFirstTime()}
      {renderConfirmDialog()}
    </div>
  );
}

export default App;
