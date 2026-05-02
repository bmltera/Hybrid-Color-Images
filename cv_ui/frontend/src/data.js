// Static data extracted from the analysis report and CSV files

export const models = [
  { name: 'ALIGN', type: 'Contrastive', overall: 0.701, grey: 0.785, color: 0.616, delta: 0.169, bias: 'Grey' },
  { name: 'SigLIP', type: 'Contrastive', overall: 0.684, grey: 0.746, color: 0.621, delta: 0.124, bias: 'Grey' },
  { name: 'LLaVA-1.6', type: 'Instruction-Tuned', overall: 0.667, grey: 0.802, color: 0.531, delta: 0.271, bias: 'Grey' },
  { name: 'SmolVLM', type: 'Compact Multimodal', overall: 0.655, grey: 0.729, color: 0.582, delta: 0.147, bias: 'Grey' },
  { name: 'Qwen2-VL', type: 'Instruction-Tuned', overall: 0.653, grey: 0.695, color: 0.610, delta: 0.085, bias: 'Grey' },
  { name: 'GPT-4o-mini', type: 'Proprietary API', overall: 0.644, grey: 0.689, color: 0.599, delta: 0.090, bias: 'Grey' },
  { name: 'LLaVA-1.5', type: 'Instruction-Tuned', overall: 0.633, grey: 0.757, color: 0.508, delta: 0.249, bias: 'Grey' },
  { name: 'CLIP', type: 'Contrastive', overall: 0.630, grey: 0.802, color: 0.458, delta: 0.345, bias: 'Grey' },
  { name: 'GPT-5.5', type: 'Proprietary API', overall: 0.540, grey: 0.497, color: 0.584, delta: -0.087, bias: 'Color' },
  { name: 'BLIP-2', type: 'Generative (Q-Former)', overall: 0.500, grey: 0.435, color: 0.565, delta: -0.130, bias: 'Color' },
  { name: 'Moondream2', type: 'Compact Generative', overall: 0.483, grey: 0.548, color: 0.418, delta: 0.130, bias: 'Grey' },
]

export const architectureRanking = [
  { name: 'Contrastive', accuracy: 0.671, models: 3, desc: 'CLIP, SigLIP, ALIGN — dual encoders with shared embedding space' },
  { name: 'Compact Multimodal', accuracy: 0.655, models: 1, desc: 'SmolVLM — Idefics3 architecture' },
  { name: 'Instruction-Tuned (Dynamic)', accuracy: 0.653, models: 1, desc: 'Qwen2-VL — dynamic resolution input' },
  { name: 'Instruction-Tuned LLM', accuracy: 0.650, models: 2, desc: 'LLaVA 1.5/1.6 — visual projection + instruction tuning' },
  { name: 'Proprietary API', accuracy: 0.592, models: 2, desc: 'GPT-4o-mini, GPT-5.5 — closed-source models' },
  { name: 'Generative (Q-Former)', accuracy: 0.500, models: 1, desc: 'BLIP-2 — query transformer + frozen LLM' },
  { name: 'Compact Generative', accuracy: 0.483, models: 1, desc: 'Moondream2 — lightweight generative VLM' },
]

// Good examples: clear illusion, high quality, most models correct
export const goodExamples = [
  { id: '0003', grey: 'bottle', color: 'bird', quality: 'M' },
  { id: '0010', grey: 'trumpet', color: 'fox', quality: 'H' },
  { id: '0015', grey: 'guitar', color: 'frog', quality: 'H' },
  { id: '0027', grey: 'piano', color: 'snail', quality: 'H' },
]

// Bad examples: both entities visible, models struggle
export const badExamples = [
  { id: '0002', grey: 'violin', color: 'eagle', quality: 'L' },
  { id: '0100', grey: 'mushroom', color: 'manatee', quality: 'L' },
  { id: '0117', grey: 'vase', color: 'iguana', quality: 'M' },
  { id: '0196', grey: 'candle', color: 'fish', quality: 'L' },
]

export const qualityTiers = [
  { tier: 'Low (L)', accuracy: 0.607, predictions: 2212, desc: 'Easy to see through — both entities often visible' },
  { tier: 'Medium (M)', accuracy: 0.618, predictions: 1230, desc: 'Moderate illusion quality' },
  { tier: 'High (H)', accuracy: 0.668, predictions: 437, desc: 'Hard to see through — dominant entity masks the other' },
]

export const keyFindings = [
  { icon: '🏆', title: 'ALIGN wins', desc: 'Best overall accuracy at 70.1%, followed by SigLIP (68.4%)' },
  { icon: '🔍', title: 'Greyscale Bias', desc: 'Most VLMs favor luminance cues over color — avg grey accuracy 68.1% vs color 55.4%' },
  { icon: '⚡', title: 'CLIP: Biggest Gap', desc: 'CLIP shows the largest grey–color delta at 34.5 percentage points' },
  { icon: '🎨', title: 'GPT-5.5 & BLIP-2 Differ', desc: 'Only two models show a color bias — they rely more on chromatic cues' },
  { icon: '📊', title: '87 Hard Images', desc: 'Some illusions fool every model, with 3 images achieving 0% accuracy' },
  { icon: '🧪', title: 'Quality Matters', desc: 'High-quality illusions (harder for humans) are actually easier for models (66.8% vs 60.7%)' },
]
