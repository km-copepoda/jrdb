import React, {useState, useEffect } from 'react'
import FilterPanel from './components/FilterPanel'
import RaceTable from './components/RaceTable'
import FieldSelector from './components/FieldSelector'
import DownloadButton from './components/DownloadButton'
import { fetchRaces, fetchFields } from './api'

const s = {
    root: { display : 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' },
    header: {
        background: '#1a237e', color: '#fff',
        padding: '8px 16px', fontSize: '15px', fontWeight: 'bold',
        display: 'flex', alignItems: 'center', gap: '12px',
    },
    headerSub: { fontSize: '12px', color: '"9fa8da', fontWeight: 'normal' },
    body: { display: 'flex', flex: 1, overflow: 'hidden' },
    left: { flex: '1 1 0', minWidth: 0, display: 'flex', flexDirection: 'column', borderRight: '1px solid #ddd' },
    right: { width: '300px', flexShrink: 0, background: '#fafafa' },
    error: {
        padding: '12px 16px', background: '#ffebee', color: '#c62828',
        borderBottom: '1px solid #ef9a9a', fontSize: '12px',
    },
} 

export default function App() {
    const [races, setRaces] = useState([])
    const [total, setTotal] = useState(0)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [selectedRaces, setSelectedRaces] = useState(new Set())

    const [availableFields, setAvailableFields] = useState({})
    const [selectedFields, setSelectedFields] = useState(new Set())

    // フィールド一覧を起動時に取得
    useEffect(() => {
        fetchFields()
            .then(data => {
                setAvailableFields(data)
                // デフォルト選択：SEDの主要フィールド
                const defaults = new Set()
                const defaultSed = ['IDM', '確定単単人気順位', '着順', '馬体重', 'テン指数', '上がり指数']
                const defaultKyi = ['IDM', '騎手指数', '基準人気順位']
                if (data.SED) defaultSed.forEach(f => { if (data.SED.find(x => x.key === f)) defaults.add('SED.' + f) })
                if (data.KYI) defaultKyi.forEach(f => { if (data.KYI.find(x => x.key === f)) defaults.add('KYI.' + f) })
                setSelectedFields(defaults)

            })
            .catch(err => setError('フィールド一覧の取得に失敗しました： ' + err.message))
    }, [])

    const handleSearch = async (filters) => {
        setLoading(true)
        setError(null)
        try {
            const data = await fetchRaces(filters)
            setRaces(data.results)
            setTotal(data.count)
        } catch (err) {
            setError('レース一覧の取得に失敗しました: ' + err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div style={s.root}>
            <div style={s.header}>
                JRDB ML データダウンロード
                <span style={s.headerSub}>-荒れるレース予測用データセット</span>
            </div>
            
            <FilterPanel onSearch={handleSearch} loading={loading} />

            {error && <div style={s.error}>{error}</div>}

            <div style={s.body}>
                <div style={s.left}>
                    <RaceTable
                        races={races}
                        selectedRaces={selectedRaces}
                        setSelectedRaces={setSelectedRaces}
                        total={total}
                        loading={loading}
                    />
                </div>
                <div style={s.right}>
                    <FieldSelector
                        availableFields={availableFields}
                        selectedFields={selectedFields}
                        setSelectedFields={setSelectedFields}
                    />
                </div>
            </div>
            
            <DownloadButton
                selectedRaces={selectedRaces}
                selectedFields={selectedFields}
            />
        </div>
    )
}