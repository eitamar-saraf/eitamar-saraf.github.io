import type { FlowNode } from './flow';

export const GPT2_FLOW: FlowNode = {
  id: 'gpt2',
  label: 'GPT-2 (124M)',
  children: [
    { id: 'tokenizer', label: 'Tokenizer', shape: 'text → [T]' },
    { id: 'embeddings', label: 'Token + position embedding', shape: '[T, 768]' },
    {
      id: 'transformer-block',
      label: 'Transformer block',
      repeat: '× 12',
      shape: '[T, 768]',
      children: [
        { id: 'attention-norm', target: 'layer-norm', label: 'LayerNorm' },
        {
          id: 'attention',
          label: 'Causal self-attention',
          children: [
            { id: 'qkv-projection', label: 'Q, K, V projection', shape: '3 × [T, 768]' },
            { id: 'split-heads', label: 'Split into 12 heads', shape: '[12, T, 64]' },
            { id: 'attention-scores', label: 'Scores Q·Kᵀ / √64', shape: '[12, T, T]' },
            { id: 'causal-mask', label: 'Causal mask' },
            { id: 'softmax', label: 'Softmax' },
            { id: 'weighted-sum', label: 'Weighted sum of V', shape: '[12, T, 64]' },
            { id: 'merge-heads', label: 'Merge heads + project', shape: '[T, 768]' },
          ],
        },
        { id: 'attention-residual', target: 'residual', label: '⊕ Residual add' },
        { id: 'mlp-norm', target: 'layer-norm', label: 'LayerNorm' },
        {
          id: 'mlp',
          label: 'MLP',
          children: [
            { id: 'up-projection', label: 'Linear 768 → 3072', shape: '[T, 3072]' },
            { id: 'gelu', label: 'GELU' },
            { id: 'down-projection', label: 'Linear 3072 → 768', shape: '[T, 768]' },
          ],
        },
        { id: 'mlp-residual', target: 'residual', label: '⊕ Residual add' },
      ],
    },
    { id: 'final-norm', label: 'Final LayerNorm', shape: '[T, 768]' },
    { id: 'lm-head', label: 'LM head (tied)', shape: '[T, 50257]' },
    { id: 'generate', label: 'Pick a token, repeat' },
  ],
};
