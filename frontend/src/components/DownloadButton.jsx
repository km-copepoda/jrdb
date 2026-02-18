import React, { useState, useCallback } from 'react'
import { buildCsvUrl } from '../api'

const styles = {
    bar: {
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '8px 16px', background: '#f5f5f5',
        borderTop: '1px solid #ddd', flexShrink: 0,
    },
    info: { fontSize: '13px', color: '#555' },
    button: {
        padding: '8px 24px', fontSize: '14px', fontWeight: 'bold',
        color: '#fff', background: '#1976d2', border: 'none',
        borderRadius: '4px', cursor: 'pointer',
    },
    buttonDisabled: {
        padding: '8px 24px', fontSize: '14px', fontWeight: 'bold',
        color: '#fff', background: '#aaa', border: 'none',
        borderRadius: '4px', cursor: 'not-allowed',
    },
    error: { fontSize: '12px', color: '#c62828', marginLeft: '12px' },
}

export default function DownloadButton({ selectedRaces, selectedFields }) {
    const [downloading, setDownloading] = useState(false)
    const [error, setError] = useState(null)

    const raceCount = selectedRaces.size
    const fieldCount = selectedFields.size
    const disabled = raceCount === 0 || fieldCount === 0 || downloading

    const handleDownload = useCallback(async () => {
        setError(null)
        setDownloading(true)
        try {
            const url = buildCsvUrl([...selectedRaces], [...selectedFields])
            const res = await fetch(url)
            if (!res.ok) throw new Error(`HTTP ${res.status}`)
            const blob = await res.blob()
            const a = document.createElement('a')
            a.href = URL.createObjectURL(blob)
            a.download = 'jrdb_ml_data.csv'
            document.body.appendChild(a)
            a.click()
            a.remove()
            URL.revokeObjectURL(a.href)
        } catch (err) {
            setError('ダウンロード失敗: ' + err.message)
        } finally {
            setDownloading(false)
        }
    }, [selectedRaces, selectedFields])

    return (
        <div style={styles.bar}>
            <span style={styles.info}>
                レース: {raceCount}件　フィールド: {fieldCount}件
                {error && <span style={styles.error}>{error}</span>}
            </span>
            <button
                style={disabled ? styles.buttonDisabled : styles.button}
                disabled={disabled}
                onClick={handleDownload}
            >
                {downloading ? 'ダウンロード中…' : 'CSV ダウンロード'}
            </button>
        </div>
    )
}
