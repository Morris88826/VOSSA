<template>
  <div class="min-vh-100 bg-white">
    <div class="bg-white" id="demo">
      <div class="container py-5">
        <!-- Demo Header -->
        <div class="text-center mb-5">
          <h2 class="demo-title">
            VOSSA: Voiceprint Optimization for Streaming Speech Architectures
          </h2>
          <p class="demo-venue">Interspeech 2026</p>
          <p class="demo-authors">
            <span
              ><a href="https://github.com/Morris88826" target="_blank">Mu-Ruei Tseng</a
              ><sup>1</sup></span
            >
            <span
              ><a href="https://github.com/warisqr007" target="_blank">Waris Quamer</a
              ><sup>1</sup></span
            >
            <span
              ><a href="https://github.com/Ghadynasrallah" target="_blank">Ghady Nasrallah</a
              ><sup>1</sup></span
            >
            <span
              ><a
                href="https://scholar.google.com/citations?user=UnuQfEwAAAAJ&hl=en"
                target="_blank"
                >Ricardo Gutierrez-Osuna</a
              ><sup>1</sup></span
            >
          </p>
          <p class="demo-affiliations"><sup>1</sup> Texas A&amp;M University</p>
        </div>

        <!-- Abstract -->
        <div class="row justify-content-center mb-5">
          <div class="col-lg-10">
            <div class="abstract-block">
              <div class="abstract-title">Abstract</div>
              <p class="abstract-text">
                Real-time voice conversion (VC) systems commonly rely on pretrained speaker
                embeddings from automatic speaker verification (ASV) models. While effective for
                speaker discrimination, these embeddings are trained to remain stable across
                phonetic and prosodic variations within-speaker, which may conflict with frame-level
                acoustic generation in streaming constraints. To address this issue, we propose
                VOSSA (Voiceprint Optimization for Streaming Speech Architectures), a speaker
                representation framework that extracts speaker information from intermediate content
                encoder layers and aggregates using attentive statistics pooling. The embedding is
                trained jointly with VC objectives, removing the need for a separate speaker
                encoder. Across six datasets, VOSSA improves F0 dynamics and vowel-discriminative
                acoustic cues while maintaining comparable NISQA-MOS, WER, and speaker similarity.
                Perceptual tests further indicate improvements in naturalness, speaker similarity,
                intelligibility, and vibrancy.
              </p>
            </div>
          </div>
        </div>

        <!-- Model Architecture -->
        <div class="row justify-content-center mb-5">
          <div class="col-lg-10">
            <div class="figure-block">
              <div class="figure-title">Model Architecture</div>
              <div class="figure-container">
                <img
                  :src="getAssetUrl('/figures/architecture.png')"
                  alt="VOSSA Model Architecture"
                  class="figure-img"
                />
              </div>
              <p class="figure-caption">
                Training workflow for the <u>TVTSyn backbone</u>. <b>(a)</b> content encoder trained
                against HuBERT k-means pseudo-labels, and <b>(b)</b> decoder conditioned on speaker
                embedding trained with self-supervision and discriminator objectives.
                <b>(c)</b> Overview of the <u>training protocol in VOSSA</u>. The
                self-reconstruction path (bottom, purple) uses segments from the same LibriTTS
                speaker to provide fully supervised training, while the non-parallel VC path (top,
                orange) uses VoxCeleb targets to condition conversion. <b>(d)</b> Speaker embedding
                extraction from the frozen content encoder. We collect features from the last CNN
                layer and every other layer of the MHSA stack, concatenate them to form <b>H</b> ∈
                ℝ<sup><i>T</i> × <i>Ld</i></sup
                >, and apply attentive statistics pooling and an MLP to obtain a global speaker
                embedding.
              </p>
            </div>
          </div>
        </div>

        <!-- Model Selection -->
        <div class="row justify-content-center mb-5">
          <div class="col-lg-10">
            <div class="models-header">
              <h3 class="models-title">Models</h3>
              <button
                @click="addModel"
                class="btn-add-model"
                :disabled="selectedModels.length >= availableModels.length"
              >
                <span>+ Add Model</span>
              </button>
            </div>

            <div class="models-grid">
              <div v-for="(model, index) in selectedModels" :key="index" class="model-card">
                <div class="model-card-header">
                  <div class="model-label">Model {{ String.fromCharCode(65 + index) }}</div>
                  <button
                    v-if="selectedModels.length > 1"
                    @click="removeModel(index)"
                    class="btn-remove-model"
                  >
                    ×
                  </button>
                </div>
                <select class="model-select" v-model="selectedModels[index]">
                  <option
                    v-for="availModel in availableModels"
                    :key="availModel"
                    :value="availModel"
                  >
                    {{ availModel }}
                  </option>
                </select>
              </div>
            </div>
          </div>
        </div>

        <!-- Pair Selection -->
        <div class="row justify-content-center mb-5">
          <div class="col-lg-10">
            <div class="selector-panel">
              <div class="selector-group">
                <label class="selector-label">Select example:</label>
                <select v-model="selectedIndex" class="pair-select">
                  <option v-for="(pair, idx) in availablePairs" :key="pair" :value="idx">
                    Example {{ String(idx + 1).padStart(3, '0') }}
                  </option>
                </select>
              </div>
            </div>
          </div>
        </div>

        <!-- Audio Display -->
        <div class="row justify-content-center">
          <div class="col-lg-10">
            <div class="audio-section">
              <div class="audio-section-title">
                Sample {{ String(selectedIndex + 1).padStart(3, '0') }}
              </div>

              <!-- Source and Target -->
              <div class="src-tgt-row">
                <div class="audio-column src-column">
                  <div class="audio-label">Source</div>
                  <audio
                    :src="getAssetUrl(`/demo/src/${availablePairs[selectedIndex]}.wav`)"
                    controls
                  ></audio>
                </div>
                <div class="audio-column tgt-column">
                  <div class="audio-label">Target</div>
                  <audio
                    :src="getAssetUrl(`/demo/tgt/${availablePairs[selectedIndex]}.wav`)"
                    controls
                  ></audio>
                </div>
              </div>

              <!-- Separator -->
              <div class="results-separator"></div>

              <!-- Model Results -->
              <div class="results-row-container">
                <div class="results-label">Converted Results:</div>
                <div
                  class="models-audio-row"
                  :style="{ gridTemplateColumns: `repeat(${selectedModels.length}, 1fr)` }"
                >
                  <div
                    v-for="(model, modelIdx) in selectedModels"
                    :key="modelIdx"
                    class="audio-column"
                  >
                    <div class="audio-label">{{ model }}</div>
                    <audio
                      :src="getAssetUrl(`/demo/vc/${model}/${availablePairs[selectedIndex]}.wav`)"
                      controls
                    ></audio>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Citation -->
        <div class="row justify-content-center mt-5">
          <div class="col-lg-10">
            <div class="citation-section">
              <h3 class="citation-heading">Citation</h3>
              <p class="citation-pending">BibTeX will be available upon official publication at Interspeech 2026.</p>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'


const getAssetUrl = (path) => {
  const base = import.meta.env.BASE_URL || '/'
  const cleanPath = path.startsWith('/') ? path.slice(1) : path
  const cleanBase = base.endsWith('/') ? base : base + '/'
  return cleanBase + cleanPath
}

const availableModels = ['slt24', 'DarkStream', 'GenVC-small', 'TVTSyn', 'VOSSA']

const selectedModels = ref(['DarkStream', 'GenVC-small', 'TVTSyn', 'VOSSA'])

const addModel = () => {
  if (selectedModels.value.length < availableModels.length) {
    const nextModel = availableModels.find((m) => !selectedModels.value.includes(m))
    if (nextModel) selectedModels.value.push(nextModel)
  }
}

const removeModel = (index) => {
  if (selectedModels.value.length > 1) {
    selectedModels.value.splice(index, 1)
  }
}

const availablePairs = [
  'H004005',
  'H005916',
  'H006981',
  'H007745',
  'H007973',
  'H009104',
  'H009116',
  'H009191',
  'H013928',
  'H014422',
  'H015175',
  'H015740',
  'H024076',
  'H024794',
  'H025586',
]

const selectedIndex = ref(0)
</script>

<style scoped>
/* Abstract */
.abstract-block {
  background: #f6f8fa;
  border: 1px solid #e1e4e8;
  border-left: 4px solid #0969da;
  border-radius: 12px;
  padding: 28px 32px;
}

.abstract-title {
  font-size: 13px;
  font-weight: 700;
  color: #0969da;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  margin-bottom: 14px;
}

.abstract-text {
  font-size: 15px;
  color: #24292f;
  line-height: 1.75;
  margin: 0;
  letter-spacing: -0.1px;
}

/* Model Architecture Figure */
.figure-block {
  background: #ffffff;
  border: 1px solid #e1e4e8;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.figure-title {
  font-size: 18px;
  font-weight: 600;
  color: #24292f;
  margin-bottom: 16px;
  letter-spacing: -0.3px;
  padding-bottom: 12px;
  border-bottom: 2px solid #f6f8fa;
}

.figure-container {
  width: 100%;
  border-radius: 8px;
  overflow: hidden;
  background: #f6f8fa;
}

.figure-img {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 8px;
}

.figure-caption {
  font-size: 13.5px;
  color: #57606a;
  line-height: 1.7;
  margin: 16px 0 0 0;
  padding-top: 14px;
  border-top: 1px solid #e1e4e8;
}

/* Demo Header */
.demo-title {
  font-size: 36px;
  font-weight: 700;
  color: #24292f;
  letter-spacing: -1px;
  margin-bottom: 8px;
  background: linear-gradient(135deg, #24292f 0%, #0969da 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.demo-venue {
  font-size: 15px;
  font-weight: 600;
  color: #0969da;
  letter-spacing: 0.02em;
  margin: 6px 0 12px;
  text-transform: uppercase;
}

.demo-authors {
  font-size: 17px;
  font-weight: 400;
  color: #24292f;
  margin: 16px 0 8px;
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 4px 24px;
  line-height: 1.6;
}

.demo-authors span {
  white-space: nowrap;
}

.demo-authors sup {
  font-size: 11px;
  color: #0969da;
  font-weight: 600;
}

.demo-affiliations {
  font-size: 14px;
  color: #57606a;
  margin: 4px 0 0;
  letter-spacing: 0.01em;
}

.demo-affiliations sup {
  font-size: 11px;
  color: #0969da;
  font-weight: 600;
}

/* Model Selection */
.models-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.models-title {
  font-size: 20px;
  font-weight: 600;
  color: #24292f;
  margin: 0;
  letter-spacing: -0.3px;
}

.btn-add-model {
  padding: 8px 16px;
  background: #0969da;
  color: #ffffff;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  letter-spacing: -0.1px;
}

.btn-add-model:hover:not(:disabled) {
  background: #0550ae;
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(9, 105, 218, 0.3);
}

.btn-add-model:disabled {
  background: #d0d7de;
  cursor: not-allowed;
  opacity: 0.6;
}

.models-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
}

.models-grid .model-card {
  flex: 1;
  min-width: 180px;
}

.model-card {
  background: #ffffff;
  border: 1px solid #e1e4e8;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
  transition: all 0.3s ease;
}

.model-card:hover {
  border-color: #c9d1d9;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  transform: translateY(-2px);
}

.model-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.model-label {
  font-size: 11px;
  font-weight: 600;
  color: #57606a;
  letter-spacing: 0.3px;
  text-transform: uppercase;
}

.btn-remove-model {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: 1px solid #d0d7de;
  border-radius: 6px;
  color: #57606a;
  font-size: 18px;
  line-height: 1;
  cursor: pointer;
  transition: all 0.2s ease;
  padding: 0;
}

.btn-remove-model:hover {
  background: #ff4444;
  border-color: #ff4444;
  color: #ffffff;
}

.model-select {
  width: 100%;
  padding: 12px 14px;
  font-size: 15px;
  border: 1.5px solid #d0d7de;
  border-radius: 8px;
  background: #ffffff;
  color: #24292f;
  font-weight: 500;
  letter-spacing: -0.2px;
  transition: all 0.2s ease;
}

.model-select:focus {
  outline: none;
  border-color: #0969da;
  box-shadow: 0 0 0 3px rgba(9, 105, 218, 0.1);
}

.model-select:hover {
  border-color: #57606a;
}

/* Pair Selector */
.selector-panel {
  background: #ffffff;
  border: 1px solid #e1e4e8;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.selector-group {
  display: flex;
  align-items: center;
  gap: 16px;
}

.selector-label {
  font-size: 14px;
  font-weight: 600;
  color: #24292f;
  letter-spacing: -0.2px;
  white-space: nowrap;
}

.pair-select {
  flex: 1;
  max-width: 300px;
  padding: 12px 14px;
  font-size: 15px;
  border: 1.5px solid #d0d7de;
  border-radius: 8px;
  background: #ffffff;
  color: #24292f;
  font-weight: 500;
  letter-spacing: -0.2px;
  transition: all 0.2s ease;
}

.pair-select:focus {
  outline: none;
  border-color: #0969da;
  box-shadow: 0 0 0 3px rgba(9, 105, 218, 0.1);
}

.pair-select:hover {
  border-color: #57606a;
}

/* Audio Section */
.audio-section {
  background: #ffffff;
  border: 1px solid #e1e4e8;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
  transition: all 0.3s ease;
}

.audio-section:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.audio-section-title {
  font-size: 19px;
  font-weight: 600;
  color: #24292f;
  margin-bottom: 20px;
  letter-spacing: -0.4px;
  padding-bottom: 12px;
  border-bottom: 2px solid #f6f8fa;
}

.src-tgt-row {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin-bottom: 0;
}

.models-audio-row {
  display: grid;
  gap: 16px;
}

.audio-column {
  display: flex;
  flex-direction: column;
  gap: 10px;
  background: #f6f8fa;
  padding: 14px;
  border-radius: 10px;
  border: 1px solid #e1e4e8;
  transition: all 0.2s ease;
}

.audio-column:hover {
  background: #ffffff;
  border-color: #c9d1d9;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.src-column {
  border-left: 3px solid #0969da;
}

.tgt-column {
  border-left: 3px solid #1a7f37;
}

.audio-label {
  font-size: 11px;
  font-weight: 600;
  color: #57606a;
  letter-spacing: 0.3px;
  text-transform: uppercase;
}

.audio-column audio {
  width: 100%;
  height: 40px;
  outline: none;
}

.results-separator {
  height: 1px;
  background: linear-gradient(to right, transparent, #d0d7de 20%, #d0d7de 80%, transparent);
  margin: 24px 0;
  position: relative;
}

.results-separator::before {
  content: '';
  position: absolute;
  top: -4px;
  left: 50%;
  transform: translateX(-50%);
  width: 8px;
  height: 8px;
  background: #0969da;
  border-radius: 50%;
  box-shadow:
    0 0 0 3px #ffffff,
    0 0 0 4px #d0d7de;
}

.results-row-container {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.results-label {
  font-size: 12px;
  font-weight: 600;
  color: #57606a;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* Responsive */
@media (max-width: 768px) {
  .demo-title {
    font-size: 28px;
    letter-spacing: -0.7px;
  }

  .demo-subtitle {
    font-size: 15px;
  }

  .model-card {
    padding: 16px;
  }

  .selector-group {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }

  .pair-select {
    max-width: 100%;
    width: 100%;
  }

  .src-tgt-row {
    grid-template-columns: 1fr;
  }

  .models-audio-row {
    grid-template-columns: 1fr !important;
  }

  .audio-section {
    padding: 16px;
  }
}

/* Citation */
.citation-section {
  padding: 24px 0 8px;
  border-top: 1px solid #e5e7eb;
}

.citation-heading {
  font-size: 18px;
  font-weight: 600;
  color: #24292f;
  margin-bottom: 8px;
}

.citation-pending {
  font-size: 15px;
  color: #57606a;
  margin: 0;
}
</style>
