import React, { useRef, useState, useCallback } from 'react'

export default function ImageComparison({ colorSrc, greySrc, greyLabel, colorLabel, quality }) {
  const containerRef = useRef(null)
  const [pos, setPos] = useState(50)
  const [dragging, setDragging] = useState(false)

  const updatePos = useCallback((clientX) => {
    const rect = containerRef.current.getBoundingClientRect()
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width))
    setPos((x / rect.width) * 100)
  }, [])

  const onPointerDown = (e) => {
    setDragging(true)
    updatePos(e.clientX)
    containerRef.current.setPointerCapture(e.pointerId)
  }

  const onPointerMove = (e) => { if (dragging) updatePos(e.clientX) }
  const onPointerUp = () => setDragging(false)

  return (
    <div className="comparison-container">
      <div
        className="comparison-wrapper"
        ref={containerRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        <img className="color-img" src={colorSrc} alt={colorLabel || 'Color view'} loading="lazy" />
        <img
          className="grey-img"
          src={greySrc}
          alt={greyLabel || 'Greyscale view'}
          style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
        />
        <div className="comparison-slider-line" style={{ left: `${pos}%` }} />
        <div className="comparison-slider-handle" style={{ left: `${pos}%` }} />
      </div>
      <div className="comparison-labels">
        <span className="comparison-label grey-label">
          <span className="indicator" />
          Greyscale → <strong>{greyLabel}</strong>
        </span>
        <span className="comparison-label color-label">
          <span className="indicator" />
          Color → <strong>{colorLabel}</strong>
        </span>
      </div>
      {quality && (
        <div className="comparison-info">
          Illusion Quality: <strong>{quality === 'H' ? 'High' : quality === 'M' ? 'Medium' : 'Low'}</strong> — Drag the slider to reveal each entity
        </div>
      )}
    </div>
  )
}
