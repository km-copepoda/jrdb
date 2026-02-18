import React, { useState } from 'react';

const VENUES = [
  { value: '', label: '全場' },
  { value: '01', label: '01札幌' },
  { value: '02', label: '02函館' },
  { value: '03', label: '03福島' },
  { value: '04', label: '04新潟' },
  { value: '05', label: '05東京' },
  { value: '06', label: '06中山' },
  { value: '07', label: '07中京' },
  { value: '08', label: '08京都' },
  { value: '09', label: '09阪神' },
  { value: '10', label: '10小倉' },
];

const GRADES = [
  { value: '', label: '全グレード' },
  { value: '1', label: 'G1' },
  { value: '2', label: 'G2' },
  { value: '3', label: 'G3' },
  { value: '4', label: '重賞' },
  { value: '5', label: '特別' },
  { value: '6', label: 'L' },
];

const TRACKS = [
  { value: '', label: '全コース' },
  { value: '1', label: '芝' },
  { value: '2', label: 'ダート' },
  { value: '3', label: '障害' },
];

const styles = {
  container: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: '12px',
    padding: '12px 16px',
    background: '#f5f5f5',
    borderRadius: '6px',
    fontSize: '14px',
  },
  label: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    whiteSpace: 'nowrap',
  },
  input: {
    padding: '4px 8px',
    border: '1px solid #ccc',
    borderRadius: '4px',
    fontSize: '14px',
    width: '100px',
  },
  select: {
    padding: '4px 8px',
    border: '1px solid #ccc',
    borderRadius: '4px',
    fontSize: '14px',
  },
  button: {
    padding: '6px 20px',
    background: '#1976d2',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    fontSize: '14px',
    cursor: 'pointer',
  },
  buttonDisabled: {
    padding: '6px 20px',
    background: '#90caf9',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    fontSize: '14px',
    cursor: 'not-allowed',
  },
};

export default function FilterPanel({ onSearch, loading }) {
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [venue, setVenue] = useState('');
  const [grade, setGrade] = useState('');
  const [track, setTrack] = useState('');
  const [distanceMin, setDistanceMin] = useState('');
  const [distanceMax, setDistanceMax] = useState('');
  const [aretaOnly, setAretaOnly] = useState(false);

  const handleSearch = () => {
    onSearch({
      date_from: dateFrom,
      date_to: dateTo,
      venue,
      grade,
      track,
      distance_min: distanceMin,
      distance_max: distanceMax,
      areta_only: aretaOnly ? '1' : '',
      page: 1,
      page_size: 100,
    });
  };

  return (
    <div style={styles.container}>
      <label style={styles.label}>
        開催日
        <input
          style={styles.input}
          placeholder="20170101"
          value={dateFrom}
          onChange={(e) => setDateFrom(e.target.value)}
        />
        〜
        <input
          style={styles.input}
          placeholder="20170101"
          value={dateTo}
          onChange={(e) => setDateTo(e.target.value)}
        />
      </label>

      <label style={styles.label}>
        競馬場
        <select style={styles.select} value={venue} onChange={(e) => setVenue(e.target.value)}>
          {VENUES.map((v) => (
            <option key={v.value} value={v.value}>{v.label}</option>
          ))}
        </select>
      </label>

      <label style={styles.label}>
        グレード
        <select style={styles.select} value={grade} onChange={(e) => setGrade(e.target.value)}>
          {GRADES.map((g) => (
            <option key={g.value} value={g.value}>{g.label}</option>
          ))}
        </select>
      </label>

      <label style={styles.label}>
        コース
        <select style={styles.select} value={track} onChange={(e) => setTrack(e.target.value)}>
          {TRACKS.map((t) => (
            <option key={t.value} value={t.value}>{t.label}</option>
          ))}
        </select>
      </label>

      <label style={styles.label}>
        距離
        <input
          style={styles.input}
          value={distanceMin}
          onChange={(e) => setDistanceMin(e.target.value)}
        />
        〜
        <input
          style={styles.input}
          value={distanceMax}
          onChange={(e) => setDistanceMax(e.target.value)}
        />
      </label>

      <label style={styles.label}>
        <input
          type="checkbox"
          checked={aretaOnly}
          onChange={(e) => setAretaOnly(e.target.checked)}
        />
        荒れたレースのみ
      </label>

      <button
        style={loading ? styles.buttonDisabled : styles.button}
        disabled={loading}
        onClick={handleSearch}
      >
        {loading ? '検索中...' : '検索'}
      </button>
    </div>
  );
}
