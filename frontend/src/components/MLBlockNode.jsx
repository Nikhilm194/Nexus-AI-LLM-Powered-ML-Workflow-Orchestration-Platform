import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import './MLBlockNode.css';

const TYPE_COLORS = {
  ingestion: { primary: '#06b6d4', glow: 'rgba(6, 182, 212, 0.25)', bg: 'rgba(6, 182, 212, 0.08)' },
  transform: { primary: '#f59e0b', glow: 'rgba(245, 158, 11, 0.25)', bg: 'rgba(245, 158, 11, 0.08)' },
  model:     { primary: '#a855f7', glow: 'rgba(168, 85, 247, 0.25)', bg: 'rgba(168, 85, 247, 0.08)' },
};

const TYPE_ICONS = {
  ingestion: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5" />
      <path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3" />
    </svg>
  ),
  transform: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3v18" />
      <path d="M18 9l-6-6-6 6" />
      <path d="M6 15l6 6 6-6" />
    </svg>
  ),
  model: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2L2 7l10 5 10-5-10-5z" />
      <path d="M2 17l10 5 10-5" />
      <path d="M2 12l10 5 10-5" />
    </svg>
  ),
};

function MLBlockNode({ data }) {
  const colors = TYPE_COLORS[data.type] || TYPE_COLORS.model;
  const icon = TYPE_ICONS[data.type] || TYPE_ICONS.model;
  const configEntries = data.config ? Object.entries(data.config) : [];

  return (
    <div
      className="ml-block-node"
      style={{
        '--node-color': colors.primary,
        '--node-glow': colors.glow,
        '--node-bg': colors.bg,
      }}
    >
      {/* Target handle (top) */}
      <Handle
        type="target"
        position={Position.Top}
        className="ml-block-handle"
        style={{ background: colors.primary }}
      />

      {/* Header */}
      <div className="ml-block-header">
        <div className="ml-block-icon" style={{ color: colors.primary }}>
          {icon}
        </div>
        <div className="ml-block-title-group">
          <span className="ml-block-label" style={{ color: colors.primary }}>
            {data.type?.toUpperCase()}
          </span>
          <h3 className="ml-block-title">{data.label}</h3>
        </div>
        <div className="ml-block-step-badge" style={{ background: colors.bg, color: colors.primary, border: `1px solid ${colors.primary}30` }}>
          {data.stepNumber}
        </div>
      </div>

      {/* Description */}
      {data.description && (
        <p className="ml-block-description">{data.description}</p>
      )}

      {/* Config preview */}
      {configEntries.length > 0 && (
        <div className="ml-block-config">
          {configEntries.slice(0, 4).map(([key, value]) => (
            <div key={key} className="ml-block-config-row">
              <span className="ml-block-config-key">{key}</span>
              <span className="ml-block-config-value" style={{ color: colors.primary }}>
                {String(value)}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Source handle (bottom) */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="ml-block-handle"
        style={{ background: colors.primary }}
      />
    </div>
  );
}

export default memo(MLBlockNode);
