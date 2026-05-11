import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import {
  ReactFlow,
  Controls,
  MiniMap,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import MLBlockNode from './MLBlockNode';
import Chatbot from './Chatbot';
import './Dashboard.css';

/* ─── Helpers ─── */

function pipelineToFlow(pipelineSteps) {
  const NODE_SPACING_Y = 280;
  const START_X = 250;
  const START_Y = 60;

  const nodes = pipelineSteps.map((step, index) => {
    // Heuristics to determine block category (color & icon)
    const blockName = step.block ? step.block.toLowerCase() : '';
    let nodeType = 'model';
    if (blockName.includes('load') || blockName.includes('read') || blockName.includes('ingest')) {
      nodeType = 'ingestion';
    } else if (blockName.includes('impute') || blockName.includes('clean') || blockName.includes('transform') || blockName.includes('scale')) {
      nodeType = 'transform';
    }

    // Create a human readable label
    const label = step.block
      ? step.block.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
      : `Step ${index + 1}`;

    return {
      id: `step-${index + 1}`,
      type: 'mlBlock',
      position: { x: START_X, y: START_Y + index * NODE_SPACING_Y },
      data: {
        label: label,
        type: nodeType,
        description: `Execute ${step.block || 'unknown block'}`,
        config: step.params || {},
        stepNumber: index + 1,
        originalBlock: step.block,
      },
      draggable: true,
    };
  });

  const edges = [];
  for (let i = 0; i < pipelineSteps.length - 1; i++) {
    edges.push({
      id: `edge-step-${i + 1}-step-${i + 2}`,
      source: `step-${i + 1}`,
      target: `step-${i + 2}`,
      type: 'smoothstep',
      animated: true,
      style: { stroke: '#6366f1', strokeWidth: 2 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: '#6366f1',
        width: 20,
        height: 20,
      },
    });
  }

  return { nodes, edges };
}

/* ─── Component ─── */

export default function Dashboard() {
  const [userIntent, setUserIntent] = useState('');
  const [csvFile, setCsvFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [pipelineGenerated, setPipelineGenerated] = useState(false);
  const [error, setError] = useState(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState(null);

  const [metadata, setMetadata] = useState(null);
  const [selectedNodeId, setSelectedNodeId] = useState(null);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/metadata')
      .then(res => res.json())
      .then(data => setMetadata(data))
      .catch(err => console.error("Failed to load metadata", err));
  }, []);

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const fileInputRef = useRef(null);

  const nodeTypes = useMemo(() => ({ mlBlock: MLBlockNode }), []);

  const onConnect = useCallback(
    (params) =>
      setEdges((eds) =>
        addEdge(
          {
            ...params,
            type: 'smoothstep',
            animated: true,
            style: { stroke: '#6366f1', strokeWidth: 2 },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: '#6366f1',
              width: 20,
              height: 20,
            },
          },
          eds
        )
      ),
    [setEdges]
  );

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setCsvFile(file);
    }
  };

  const handleRemoveFile = () => {
    setCsvFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const updateNodeConfig = (nodeId, key, value) => {
    setNodes((nds) => nds.map(node => {
      if (node.id === nodeId) {
        return {
          ...node,
          data: {
            ...node.data,
            config: { ...node.data.config, [key]: value }
          }
        };
      }
      return node;
    }));
  };

  const handleGenerate = async () => {
    if (!userIntent.trim()) {
      setError('Please describe your ML intent first.');
      return;
    }

    if (!csvFile) {
      setError('Please upload a CSV dataset.');
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      // 1. Profile CSV
      const formData = new FormData();
      formData.append('file', csvFile);

      const profileResponse = await fetch('http://127.0.0.1:8000/api/profile', {
        method: 'POST',
        body: formData,
      });

      if (!profileResponse.ok) {
        throw new Error('Failed to profile dataset.');
      }

      const profileData = await profileResponse.json();

      // 2. Generate Pipeline
      const generateResponse = await fetch('http://127.0.0.1:8000/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_intent: userIntent,
          data_profile: profileData.profile,
        }),
      });

      if (!generateResponse.ok) {
        throw new Error('Failed to generate pipeline.');
      }

      const pipelineData = await generateResponse.json();

      // Handle the case where the API returns { pipeline: [...] } instead of directly [...]
      const pipelineSteps = Array.isArray(pipelineData) ? pipelineData : (pipelineData.pipeline || []);

      const { nodes: newNodes, edges: newEdges } = pipelineToFlow(pipelineSteps);
      setNodes(newNodes);
      setEdges(newEdges);
      setPipelineGenerated(true);
    } catch (err) {
      setError(err.message || 'Failed to generate pipeline. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleExecute = async () => {
    if (nodes.length === 0) return;
    if (!csvFile) {
      setError('Please upload a CSV dataset before running the pipeline.');
      return;
    }

    setIsExecuting(true);
    setExecutionResult(null);

    try {
      const pipelineJson = nodes.map(node => ({
        block: node.data.originalBlock,
        params: node.data.config,
      }));

      const formData = new FormData();
      formData.append('file', csvFile);
      formData.append('pipeline_json', JSON.stringify(pipelineJson));

      const response = await fetch('http://127.0.0.1:8000/api/execute', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Execution failed: ${response.statusText}`);
      }

      const result = await response.json();
      setExecutionResult(result);
    } catch (err) {
      setExecutionResult({ status: 'error', detail: err.message });
      console.error(err);
    } finally {
      setIsExecuting(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleGenerate();
    }
  };

  return (
    <div className="dashboard">
      {/* ─── Sidebar ─── */}
      <aside className="dashboard-sidebar">
        {/* Logo */}
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <div>
            <h1 className="sidebar-logo-title">Nexus</h1>
            <p className="sidebar-logo-subtitle">ML Pipeline Builder</p>
          </div>
        </div>

        <div className="sidebar-divider" />

        {/* Intent Input */}
        <div className="sidebar-section">
          <label className="sidebar-label" htmlFor="user-intent">
            <span className="sidebar-label-icon">⚡</span>
            User Intent
          </label>
          <textarea
            id="user-intent"
            className="sidebar-textarea"
            placeholder="e.g. Classify customer churn using historical data..."
            value={userIntent}
            onChange={(e) => setUserIntent(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={4}
          />
        </div>

        {/* File Upload */}
        <div className="sidebar-section">
          <label className="sidebar-label">
            <span className="sidebar-label-icon">📄</span>
            Dataset (CSV)
          </label>
          <div
            className={`sidebar-upload ${csvFile ? 'has-file' : ''}`}
            onClick={() => !csvFile && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="sidebar-upload-input"
              id="csv-upload"
            />
            {csvFile ? (
              <div className="sidebar-upload-file">
                <div className="sidebar-upload-file-info">
                  <svg className="sidebar-upload-file-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                    <polyline points="14,2 14,8 20,8" />
                  </svg>
                  <div>
                    <p className="sidebar-upload-file-name">{csvFile.name}</p>
                    <p className="sidebar-upload-file-size">
                      {(csvFile.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>
                <button className="sidebar-upload-remove" onClick={handleRemoveFile} title="Remove file">
                  ✕
                </button>
              </div>
            ) : (
              <div className="sidebar-upload-placeholder">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                  <polyline points="17,8 12,3 7,8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                <span>Click to upload CSV</span>
                <span className="sidebar-upload-hint">or drag & drop</span>
              </div>
            )}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="sidebar-error">
            <span>⚠️</span>
            {error}
          </div>
        )}

        {/* Generate Button */}
        <button
          className={`sidebar-generate-btn ${isLoading ? 'loading' : ''}`}
          onClick={handleGenerate}
          disabled={isLoading}
          id="generate-pipeline-btn"
        >
          {isLoading ? (
            <>
              <div className="sidebar-generate-spinner" />
              Generating Pipeline...
            </>
          ) : (
            <>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="13,2 3,14 12,14 11,22 21,10 12,10" />
              </svg>
              Generate Pipeline
            </>
          )}
        </button>

        {/* Status badge */}
        {pipelineGenerated && !isLoading && (
          <div className="sidebar-status">
            <div className="sidebar-status-dot" />
            Pipeline ready — {nodes.length} blocks generated
          </div>
        )}

        {/* Footer */}
        <div className="sidebar-footer">
          <p>Drag nodes to rearrange • Connect ports to build custom flows</p>
        </div>
      </aside>

      {/* ─── Canvas ─── */}
      <main className="dashboard-canvas">
        {nodes.length === 0 && !isLoading ? (
          <div className="canvas-empty">
            <div className="canvas-empty-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2L2 7l10 5 10-5-10-5z" />
                <path d="M2 17l10 5 10-5" />
                <path d="M2 12l10 5 10-5" />
              </svg>
            </div>
            <h2 className="canvas-empty-title">No Pipeline Yet</h2>
            <p className="canvas-empty-text">
              Describe your ML intent in the sidebar and click{' '}
              <strong>Generate Pipeline</strong> to visualize your workflow.
            </p>
          </div>
        ) : (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={(event, node) => setSelectedNodeId(node.id)}
            onPaneClick={() => setSelectedNodeId(null)}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.3 }}
            proOptions={{ hideAttribution: true }}
            defaultEdgeOptions={{
              type: 'smoothstep',
              animated: true,
            }}
          >
            <Controls
              className="flow-controls"
              showInteractive={false}
            />
            <MiniMap
              className="flow-minimap"
              nodeColor={(node) => {
                const type = node.data?.type;
                if (type === 'ingestion') return '#06b6d4';
                if (type === 'transform') return '#f59e0b';
                return '#a855f7';
              }}
              maskColor="rgba(10, 10, 15, 0.8)"
              style={{ background: '#12121a', border: '1px solid rgba(99, 102, 241, 0.15)' }}
            />
            <Background
              color="rgba(99, 102, 241, 0.08)"
              gap={24}
              size={1}
            />
          </ReactFlow>
        )}

        {nodes.length > 0 && (
          <button
            className={`canvas-execute-btn ${isExecuting ? 'executing' : ''}`}
            onClick={handleExecute}
            disabled={isExecuting}
          >
            {isExecuting ? 'Running Pipeline...' : '▶ Run Pipeline'}
          </button>
        )}

        {/* Console / Results Area */}
        <div className={`canvas-console ${executionResult ? 'open' : ''}`}>
          <div className="console-header">
            <span>Console / Execution Results</span>
            {executionResult && (
              <button className="console-close" onClick={() => setExecutionResult(null)}>✕</button>
            )}
          </div>
          <div className="console-body">
            {executionResult ? (
              <div className="console-result-content">
                {executionResult.artifacts && Object.keys(executionResult.artifacts).length > 0 && (
                  <div className="console-artifacts">
                    <h4>💾 Generated Artifacts</h4>
                    <div className="artifact-links">
                      {executionResult.artifacts.model_url && (
                        <a
                          href={`http://127.0.0.1:8000${executionResult.artifacts.model_url}`}
                          download
                          className="artifact-btn"
                          target="_blank"
                          rel="noreferrer"
                        >
                          Download Model (.pkl)
                        </a>
                      )}
                      {executionResult.artifacts.dataframe_url && (
                        <a
                          href={`http://127.0.0.1:8000${executionResult.artifacts.dataframe_url}`}
                          download
                          className="artifact-btn"
                          target="_blank"
                          rel="noreferrer"
                        >
                          Download Processed Data (.csv)
                        </a>
                      )}
                    </div>
                  </div>
                )}
                <pre className="console-pre">{JSON.stringify(executionResult, null, 2)}</pre>
              </div>
            ) : (
              <div className="console-empty">Awaiting execution...</div>
            )}
          </div>
        </div>

        {/* Node Configuration Panel */}
        <aside className={`dashboard-config-panel ${selectedNodeId ? 'open' : ''}`}>
          <div className="config-header">
            <h3>Node Configuration</h3>
            <button className="config-close" onClick={() => setSelectedNodeId(null)}>✕</button>
          </div>
          <div className="config-body">
            {(() => {
              const selectedNode = nodes.find(n => n.id === selectedNodeId);
              const selectedBlockMeta = selectedNode && metadata?.blocks ? metadata.blocks[selectedNode.data.originalBlock] : null;

              if (!selectedNode) return <div className="config-empty">Select a node to configure.</div>;

              return (
                <>
                  <div className="config-block-title">
                    <h4>{selectedNode.data.label}</h4>
                    <span className="config-block-type">{selectedNode.data.type}</span>
                  </div>

                  {selectedBlockMeta && selectedBlockMeta.parameters && Object.keys(selectedBlockMeta.parameters).length > 0 ? (
                    <div className="config-form">
                      {Object.entries(selectedBlockMeta.parameters).map(([paramName, paramMeta]) => {
                        const currentValue = selectedNode.data.config[paramName] ?? paramMeta.default ?? '';

                        return (
                          <div key={paramName} className="config-form-group">
                            <label title={paramMeta.description}>
                              {paramName} {paramMeta.required && <span className="required">*</span>}
                            </label>
                            {paramMeta.allowed_values ? (
                              <select
                                value={currentValue}
                                onChange={(e) => updateNodeConfig(selectedNode.id, paramName, e.target.value)}
                                className="config-select"
                              >
                                <option value="" disabled>Select {paramName}...</option>
                                {paramMeta.allowed_values.map(val => (
                                  <option key={val} value={val}>{val}</option>
                                ))}
                              </select>
                            ) : (
                              <input
                                type="text"
                                value={currentValue}
                                onChange={(e) => updateNodeConfig(selectedNode.id, paramName, e.target.value)}
                                className="config-input"
                                placeholder={`Enter ${paramName}...`}
                              />
                            )}
                            <p className="config-help">{paramMeta.description}</p>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="config-empty">No configurable parameters for this block.</div>
                  )}
                </>
              );
            })()}
          </div>
        </aside>
      </main>

      {/* ─── Chatbot ─── */}
      <Chatbot />
    </div>
  );
}
