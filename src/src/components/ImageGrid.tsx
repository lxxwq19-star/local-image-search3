import { useState } from 'react';
import { convertFileSrc, invoke } from '@tauri-apps/api/core';

interface ImageGridProps {
  images: Array<{
    id: number;
    path: string;
    similarity?: number;
    width?: number;
    height?: number;
  }>;
  loading?: boolean;
}

function ImageGrid({ images, loading }: ImageGridProps) {
  const [imageErrors, setImageErrors] = useState<Set<number>>(new Set());

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
        <p>加载图片中...</p>
      </div>
    );
  }

  if (images.length === 0) {
    return null;
  }

  const handleImageClick = (path: string) => {
    invoke('open_file', { path }).catch(err => console.error('打开文件失败:', err));
  };

  const handleImgError = (id: number) => {
    setImageErrors(prev => new Set(prev).add(id));
  };

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
      gap: '14px',
      marginTop: '20px',
    }}>
      {images.map((image) => {
        const fileName = image.path.split(/[/\\]/).pop() || '未知图片';
        const hasError = imageErrors.has(image.id);

        return (
          <div
            key={image.id}
            style={{
              border: '1px solid #e0e0e0',
              borderRadius: '8px',
              overflow: 'hidden',
              backgroundColor: '#fafafa',
              cursor: 'pointer',
              transition: 'transform 0.15s, box-shadow 0.2s',
            }}
            onClick={() => handleImageClick(image.path)}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 4px 16px rgba(0,0,0,0.12)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            {/* 缩略图 */}
            <div style={{ width: '100%', height: '160px', position: 'relative', backgroundColor: '#f0f0f0' }}>
              {!hasError ? (
                <img
                  src={convertFileSrc(image.path)}
                  alt={fileName}
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    display: 'block',
                  }}
                  onError={() => handleImgError(image.id)}
                  loading="lazy"
                />
              ) : (
                <div style={{
                  width: '100%',
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#bbb',
                  fontSize: '32px',
                }}>
                  ❌
                </div>
              )}

              {/* 相似度角标 */}
              {image.similarity !== undefined && (
                <span style={{
                  position: 'absolute',
                  top: '6px',
                  right: '6px',
                  backgroundColor: image.similarity > 0.8
                    ? 'rgba(76,175,80,0.9)'
                    : image.similarity > 0.5
                    ? 'rgba(255,152,0,0.9)'
                    : 'rgba(158,158,158,0.9)',
                  color: 'white',
                  fontSize: '11px',
                  fontWeight: 600,
                  padding: '2px 7px',
                  borderRadius: '10px',
                }}>
                  {(image.similarity * 100).toFixed(0)}%
                </span>
              )}
            </div>

            {/* 文件信息 */}
            <div style={{ padding: '8px' }}>
              <p style={{
                fontSize: '12px',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                margin: 0,
                color: '#333',
                lineHeight: '1.4',
              }} title={fileName}>
                {fileName}
              </p>
              {image.width && image.height && (
                <p style={{ fontSize: '10px', color: '#999', margin: '4px 0 0 0' }}>
                  {image.width} × {image.height}
                </p>
              )}
            </div>

            {/* 点击提示 */}
            <div style={{
              textAlign: 'center',
              padding: '4px 8px',
              fontSize: '10px',
              color: '#aaa',
              borderTop: '1px solid #eee',
              backgroundColor: '#fafafa',
            }}>
              点击打开原图
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default ImageGrid;
