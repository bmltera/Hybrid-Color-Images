import React from 'react'
import FadeUp from './FadeUp'
import ImageComparison from './ImageComparison'
import {
  models, architectureRanking, goodExamples, badExamples,
  qualityTiers, keyFindings
} from './data'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, Legend
} from 'recharts'

// Use public paths (served via symlinks)
const chartPath = (name) => `/charts/${name}`
const reportPath = (name) => `/reports/${name}`
const img = (id, type) => `/images/${id}${type}.png`

const GREY_COLOR = '#94a3b8'
const COLOR_COLOR = '#c084fc'

function Nav() {
  return (
    <nav className="nav">
      <div className="nav-inner">
        <div className="nav-logo">Color Hybrids × VLMs</div>
        <ul className="nav-links">
          <li><a href="#overview">Overview</a></li>
          <li><a href="#illusions">Illusions</a></li>
          <li><a href="#results">Results</a></li>
          <li><a href="#findings">Findings</a></li>
          <li><a href="https://huggingface.co/datasets/bmltera/color-hybrid-illusions" target="_blank" rel="noopener noreferrer" className="nav-dataset">Dataset ↗</a></li>
        </ul>
      </div>
    </nav>
  )
}

function Hero() {
  return (
    <section className="hero">
      <div className="hero-content">
        <FadeUp>
          <div className="hero-badge"><span className="dot" /> Computer Vision Research</div>
        </FadeUp>
        <FadeUp delay={100}>
          <h1>
            Can AI See Through<br />
            <span className="gradient-text">Color Illusions?</span>
          </h1>
        </FadeUp>
        <FadeUp delay={200}>
          <p className="hero-subtitle">
            We tested 11 vision-language models on images where color and greyscale
            tell different stories. The results reveal a systematic bias in how AI sees the world.
          </p>
        </FadeUp>
        <FadeUp delay={300}>
          <div className="hero-authors">
            <span className="author-chip">Bill Li</span>
            <span className="author-chip">Paul Junver Soriano</span>
            <span className="author-chip">Rahul Koonantavida</span>
          </div>
        </FadeUp>
        <FadeUp delay={400}>
          <div className="hero-stats">
            <div className="hero-stat">
              <div className="hero-stat-value">11</div>
              <div className="hero-stat-label">Models Tested</div>
            </div>
            <div className="hero-stat">
              <div className="hero-stat-value">177</div>
              <div className="hero-stat-label">Image Pairs</div>
            </div>
          </div>
        </FadeUp>
      </div>
      <div className="scroll-indicator">
        <span>Scroll</span>
        <div className="arrow" />
      </div>
    </section>
  )
}

function Overview() {
  return (
    <section className="section" id="overview">
      <FadeUp>
        <div className="section-label">01 — The Problem</div>
        <h2>What Are Color Hybrid Illusions?</h2>
        <p className="section-desc">
          Using <strong>Factorized Diffusion</strong>, we generate images that depict
          one entity in color and a completely different entity in greyscale — all within
          a single image. When you remove color, a violin appears. Add color back, and an
          eagle emerges.
        </p>
      </FadeUp>
      <div className="explainer-grid">
        {[
          { n: '01', title: 'Generate Hybrid Images', desc: 'Factorized Diffusion decomposes images into color and greyscale components, conditioning each on a different prompt to produce illusions.' },
          { n: '02', title: 'Feed to VLMs', desc: 'We show both the color and greyscale versions to 11 different vision-language models and ask them to identify the entity.' },
          { n: '03', title: 'Measure Bias', desc: 'By comparing greyscale vs. color accuracy, we reveal whether each model relies more on luminance (structure) or chromatic (color) cues.' },
        ].map((c, i) => (
          <FadeUp key={i} delay={i * 100}>
            <div className="explainer-card">
              <span className="card-number">{c.n}</span>
              <h3>{c.title}</h3>
              <p>{c.desc}</p>
            </div>
          </FadeUp>
        ))}
      </div>
    </section>
  )
}

function ShowcaseCard({ id, grey, color, quality }) {
  const containerRef = React.useRef(null)
  const [pos, setPos] = React.useState(50)
  const [dragging, setDragging] = React.useState(false)

  const updatePos = React.useCallback((clientX) => {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width))
    setPos((x / rect.width) * 100)
  }, [])

  return (
    <div className="showcase-item">
      <div
        className="showcase-image-wrap"
        ref={containerRef}
        onPointerDown={(e) => { setDragging(true); updatePos(e.clientX); containerRef.current.setPointerCapture(e.pointerId) }}
        onPointerMove={(e) => dragging && updatePos(e.clientX)}
        onPointerUp={() => setDragging(false)}
      >
        <img className="color-img" src={img(id, 'c')} alt={color} loading="lazy" />
        <img
          className="grey-img"
          src={img(id, 'g')}
          alt={grey}
          style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
          loading="lazy"
        />
        <div className="comparison-slider-line" style={{ left: `${pos}%` }} />
        <div className="comparison-slider-handle" style={{ left: `${pos}%` }} />
      </div>
      <div className="showcase-meta">
        <h4>Pair #{id}</h4>
        <div className="entity-tags">
          <span className="entity-tag grey">Grey: {grey}</span>
          <span className="entity-tag color">Color: {color}</span>
          <span className="entity-tag quality">Quality: {quality}</span>
        </div>
      </div>
    </div>
  )
}

function IllusionShowcase() {
  return (
    <section className="section" id="illusions">
      <FadeUp>
        <div className="section-label">02 — The Dataset</div>
        <h2>Explore the Illusions</h2>
        <p className="section-desc">
          Drag the slider on each image to reveal the hidden entity. In color, you see
          one thing — strip away color and something entirely different appears.
        </p>
      </FadeUp>

      <FadeUp>
        <ImageComparison
          colorSrc={img('0010', 'c')}
          greySrc={img('0010', 'g')}
          greyLabel="Fan"
          colorLabel="Chameleon"
          quality="H"
        />
      </FadeUp>

      <div className="pair-section">
        <FadeUp>
          <h3>✨ High-Quality Illusions</h3>
          <p className="pair-desc">
            These illusions effectively hide one entity — models generally identify the correct
            entity for each view.
          </p>
        </FadeUp>
        <div className="showcase-grid">
          {goodExamples.map((ex, i) => (
            <FadeUp key={ex.id} delay={i * 80}>
              <ShowcaseCard {...ex} />
            </FadeUp>
          ))}
        </div>
      </div>

      <div className="pair-section">
        <FadeUp>
          <h3>💀 Tricky Pairs — Models Struggle</h3>
          <p className="pair-desc">
            In these images, both entities bleed through in both views, confusing nearly all models.
          </p>
        </FadeUp>
        <div className="showcase-grid">
          {badExamples.map((ex, i) => (
            <FadeUp key={ex.id} delay={i * 80}>
              <ShowcaseCard {...ex} />
            </FadeUp>
          ))}
        </div>
      </div>
    </section>
  )
}

function ResultsSection() {
  const chartData = models.map(m => ({
    name: m.name,
    grey: +(m.grey * 100).toFixed(1),
    color: +(m.color * 100).toFixed(1),
  }))

  return (
    <section className="section" id="results">
      <FadeUp>
        <div className="section-label">03 — Results</div>
        <h2>Model Performance</h2>
        <p className="section-desc">
          Across 11 models, greyscale recognition consistently outperforms color
          recognition — revealing a systematic luminance bias in VLMs.
        </p>
      </FadeUp>

      <FadeUp>
        <div className="custom-chart-container">
          <h3>Grey vs Color Accuracy by Model (%)</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 60 }}>
              <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} angle={-35} textAnchor="end" interval={0} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0, 100]} />
              <Tooltip
                contentStyle={{ background: '#1c1c28', border: '1px solid rgba(148,163,184,0.15)', borderRadius: 8, color: '#f1f5f9' }}
                labelStyle={{ color: '#a78bfa', fontWeight: 600 }}
              />
              <Legend wrapperStyle={{ paddingTop: 10 }} />
              <Bar dataKey="grey" name="Greyscale" fill={GREY_COLOR} radius={[4, 4, 0, 0]} />
              <Bar dataKey="color" name="Color" fill={COLOR_COLOR} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </FadeUp>

      <FadeUp>
        <div className="finding-callout">
          <h3>📌 Key Insight: The Greyscale Advantage</h3>
          <p>
            On average, models achieve <strong>68.1%</strong> accuracy on greyscale images vs
            only <strong>55.4%</strong> on color images — a <strong>12.7 percentage point gap</strong>.
            This means VLMs rely more on structural/luminance information than color when
            both are present but conflicting.
          </p>
        </div>
      </FadeUp>

      <FadeUp>
        <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.3rem', marginBottom: '1.5rem' }}>
          Per-Model Breakdown
        </h3>
      </FadeUp>
      <div className="model-grid">
        {models.map((m, i) => (
          <FadeUp key={m.name} delay={i * 60}>
            <div className="model-card">
              <div className="model-card-header">
                <div className="model-card-name">{m.name}</div>
                <div className="model-card-type">{m.type}</div>
              </div>
              <div className={`model-card-accuracy ${i === 0 ? 'top' : ''}`}>
                {(m.overall * 100).toFixed(1)}%
              </div>
              <div className="model-card-bars">
                <div className="bar-row">
                  <span className="bar-label">Grey</span>
                  <div className="bar-track">
                    <div className="bar-fill grey" style={{ width: `${m.grey * 100}%` }} />
                  </div>
                  <span className="bar-value">{(m.grey * 100).toFixed(0)}%</span>
                </div>
                <div className="bar-row">
                  <span className="bar-label">Color</span>
                  <div className="bar-track">
                    <div className="bar-fill color" style={{ width: `${m.color * 100}%` }} />
                  </div>
                  <span className="bar-value">{(m.color * 100).toFixed(0)}%</span>
                </div>
              </div>
              <div className={`bias-badge ${m.bias === 'Grey' ? 'grey-bias' : 'color-bias'}`}>
                {m.bias === 'Grey' ? '◐' : '◑'} {m.bias} bias ({m.delta > 0 ? '+' : ''}{(m.delta * 100).toFixed(1)}pp)
              </div>
            </div>
          </FadeUp>
        ))}
      </div>

      <FadeUp>
        <div className="chart-container" style={{ marginTop: '3rem' }}>
          <img src={chartPath('cross_model_grey_vs_color_delta.png')} alt="Grey vs Color delta across models" />
          <div className="chart-caption">Figure 1. Grey–Color accuracy delta across all models. Positive = greyscale bias.</div>
        </div>
      </FadeUp>
    </section>
  )
}

function ArchitectureSection() {
  return (
    <section className="section">
      <FadeUp>
        <div className="section-label">04 — Architecture Analysis</div>
        <h2>Which Architecture Wins?</h2>
        <p className="section-desc">
          Contrastive models (dual image + text encoders) outperform generative
          and instruction-tuned architectures on this benchmark.
        </p>
      </FadeUp>
      <div className="arch-timeline">
        {architectureRanking.map((a, i) => (
          <FadeUp key={a.name} delay={i * 80}>
            <div className="arch-item">
              <div className="arch-rank">#{i + 1}</div>
              <div className="arch-info">
                <div className="arch-name">{a.name}</div>
                <div className="arch-desc">{a.desc}</div>
              </div>
              <div className="arch-accuracy">{(a.accuracy * 100).toFixed(1)}%</div>
            </div>
          </FadeUp>
        ))}
      </div>

      <FadeUp>
        <div className="chart-container" style={{ marginTop: '3rem' }}>
          <img src={reportPath('model_performance_radar.png')} alt="Model performance radar chart" />
          <div className="chart-caption">Figure 2. Multi-dimensional model performance comparison.</div>
        </div>
      </FadeUp>
    </section>
  )
}

function QualitySection() {
  const data = qualityTiers.map(q => ({
    name: q.tier,
    accuracy: +(q.accuracy * 100).toFixed(1),
    predictions: q.predictions,
  }))

  return (
    <section className="section">
      <FadeUp>
        <div className="section-label">05 — Illusion Quality</div>
        <h2>Does Illusion Quality Matter?</h2>
        <p className="section-desc">
          Counter-intuitively, higher-quality illusions (harder for humans to see through)
          are actually <em>easier</em> for models — because the dominant entity is cleaner.
        </p>
      </FadeUp>

      <FadeUp>
        <div className="custom-chart-container">
          <h3>Accuracy by Illusion Quality Tier (%)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
              <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 13 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[50, 75]} />
              <Tooltip
                contentStyle={{ background: '#1c1c28', border: '1px solid rgba(148,163,184,0.15)', borderRadius: 8, color: '#f1f5f9' }}
              />
              <Bar dataKey="accuracy" name="Accuracy (%)" radius={[6, 6, 0, 0]}>
                {data.map((_, i) => (
                  <Cell key={i} fill={['#64748b', '#8b5cf6', '#6366f1'][i]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </FadeUp>

      <FadeUp>
        <div className="finding-callout">
          <h3>🧪 Surprising Result</h3>
          <p>
            High-quality illusions achieve <strong>66.8%</strong> model accuracy vs
            only <strong>60.7%</strong> for low-quality ones. When the illusion is
            well-crafted, the intended entity dominates each view more cleanly,
            making it easier for VLMs despite being harder for human perception.
          </p>
        </div>
      </FadeUp>
    </section>
  )
}

function FindingsSection() {
  return (
    <section className="section" id="findings">
      <FadeUp>
        <div className="section-label">06 — Key Findings</div>
        <h2>What We Learned</h2>
        <p className="section-desc">
          Our evaluation reveals systematic patterns in how VLMs process
          conflicting visual cues in hybrid color images.
        </p>
      </FadeUp>
      <div className="conclusion-grid">
        {keyFindings.map((f, i) => (
          <FadeUp key={i} delay={i * 80}>
            <div className="conclusion-card">
              <div className="conclusion-icon">{f.icon}</div>
              <h4>{f.title}</h4>
              <p>{f.desc}</p>
            </div>
          </FadeUp>
        ))}
      </div>

      <FadeUp>
        <div className="chart-container" style={{ marginTop: '3rem' }}>
          <img src={reportPath('model_ranking_heatmap.png')} alt="Model ranking heatmap" />
          <div className="chart-caption">Figure 3. Model ranking heatmap across evaluation dimensions.</div>
        </div>
      </FadeUp>
    </section>
  )
}

function Footer() {
  const refs = [
    'Geng et al. Visual Anagrams (CVPR 2024)',
    'Geng et al. Factorized Diffusion (ECCV 2024)',
    'Chen et al. MMStar (NeurIPS 2024)',
    'Hou et al. VIA-Bench (2026)',
    'Hessel et al. CLIPScore (EMNLP 2021)',
    'Radford et al. CLIP (ICML 2021)',
    'Li et al. BLIP-2 (ICML 2023)',
    'Liu et al. LLaVA (NeurIPS 2023)',
  ]

  return (
    <footer className="footer">
      <p>
        Entity Recognition with Vision Language Models on Diffusion-Based Color Hybrid Illusions
      </p>
      <p style={{ marginTop: '0.5rem', color: 'var(--text-muted)' }}>
        Bill Li · Paul Junver Soriano · Rahul Koonantavida
      </p>
      <div className="refs">
        {refs.map((r, i) => (
          <span key={i} className="ref-chip">[{i + 1}] {r}</span>
        ))}
      </div>
    </footer>
  )
}

export default function App() {
  return (
    <>
      <Nav />
      <Hero />
      <div className="divider" />
      <Overview />
      <div className="divider" />
      <IllusionShowcase />
      <div className="divider" />
      <ResultsSection />
      <div className="divider" />
      <ArchitectureSection />
      <div className="divider" />
      <QualitySection />
      <div className="divider" />
      <FindingsSection />
      <Footer />
    </>
  )
}
