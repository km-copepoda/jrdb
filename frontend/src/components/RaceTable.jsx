import React, { useCallback } from "react";

const VENUE_MAP = {
  "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
  "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
};

const GRADE_MAP = {
  A: "G1", B: "G2", C: "G3",
};

const TRACK_MAP = {
  1: "芝", 2: "ダ", 3: "障",
};

function formatDate(yyyymmdd) {
  if (!yyyymmdd || yyyymmdd.length !== 8) return yyyymmdd;
  return `${yyyymmdd.slice(0, 4)}/${yyyymmdd.slice(4, 6)}/${yyyymmdd.slice(6, 8)}`;
}

const styles = {
  container: {
    fontFamily: "'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
    fontSize: 14,
  },
  toolbar: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "8px 12px",
    background: "#f5f5f5",
    borderBottom: "1px solid #ddd",
    flexWrap: "wrap",
  },
  btn: {
    padding: "4px 10px",
    fontSize: 13,
    border: "1px solid #bbb",
    borderRadius: 4,
    background: "#fff",
    cursor: "pointer",
  },
  info: {
    marginLeft: "auto",
    fontSize: 13,
    color: "#555",
  },
  scrollArea: {
    maxHeight: 520,
    overflow: "auto",
    border: "1px solid #ddd",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    tableLayout: "fixed",
  },
  th: {
    position: "sticky",
    top: 0,
    background: "#e8e8e8",
    padding: "6px 8px",
    borderBottom: "2px solid #ccc",
    textAlign: "left",
    fontSize: 13,
    whiteSpace: "nowrap",
  },
  td: {
    padding: "5px 8px",
    borderBottom: "1px solid #eee",
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
  },
  badge: {
    display: "inline-block",
    background: "#e53e3e",
    color: "#fff",
    fontSize: 11,
    fontWeight: "bold",
    padding: "1px 6px",
    borderRadius: 4,
    marginLeft: 4,
  },
};

export default function RaceTable({
  races = [],
  selectedRaces,
  setSelectedRaces,
  total = 0,
  loading = false,
}) {
  const toggleOne = useCallback(
    (id) => {
      setSelectedRaces((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      });
    },
    [setSelectedRaces]
  );

  const selectAllPage = useCallback(() => {
    setSelectedRaces((prev) => {
      const next = new Set(prev);
      races.forEach((r) => next.add(r.id));
      return next;
    });
  }, [setSelectedRaces, races]);

  const deselectAllPage = useCallback(() => {
    setSelectedRaces((prev) => {
      const next = new Set(prev);
      races.forEach((r) => next.delete(r.id));
      return next;
    });
  }, [setSelectedRaces, races]);

  const selectAreta = useCallback(() => {
    setSelectedRaces((prev) => {
      const next = new Set(prev);
      races.forEach((r) => {
        if (r.areta) next.add(r.id);
      });
      return next;
    });
  }, [setSelectedRaces, races]);

  const clearAll = useCallback(() => {
    setSelectedRaces(new Set());
  }, [setSelectedRaces]);

  const allPageSelected =
    races.length > 0 && races.every((r) => selectedRaces.has(r.id));

  return (
    <div style={styles.container}>
      {/* Toolbar */}
      <div style={styles.toolbar}>
        <button
          style={styles.btn}
          onClick={allPageSelected ? deselectAllPage : selectAllPage}
        >
          {allPageSelected ? "ページ全解除" : "ページ全選択"}
        </button>
        <button style={styles.btn} onClick={selectAreta}>
          荒れたを追加選択
        </button>
        <button style={styles.btn} onClick={clearAll}>
          全解除
        </button>
        <span style={styles.info}>
          {loading ? (
            "読み込み中..."
          ) : (
            <>選択: {selectedRaces.size}件 / 全 {total}件</>
          )}
        </span>
      </div>

      {/* Scrollable table */}
      <div style={styles.scrollArea}>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={{ ...styles.th, width: 36, textAlign: "center" }}></th>
              <th style={{ ...styles.th, width: 90 }}>日付</th>
              <th style={{ ...styles.th, width: 56 }}>場</th>
              <th style={{ ...styles.th, width: 36, textAlign: "center" }}>R</th>
              <th style={{ ...styles.th, width: 36 }}>コース</th>
              <th style={{ ...styles.th, width: 56, textAlign: "right" }}>距離</th>
              <th style={{ ...styles.th, width: 48 }}>格</th>
              <th style={{ ...styles.th, width: 48, textAlign: "right" }}>頭数</th>
              <th style={styles.th}>レース名</th>
            </tr>
          </thead>
          <tbody>
            {races.map((r) => {
              const selected = selectedRaces.has(r.id);
              const rowBg = r.areta
                ? "rgba(229,62,62,0.08)"
                : selected
                ? "rgba(66,153,225,0.10)"
                : undefined;
              return (
                <tr
                  key={r.id}
                  style={{ background: rowBg, cursor: "pointer" }}
                  onClick={() => toggleOne(r.id)}
                >
                  <td style={{ ...styles.td, textAlign: "center" }}>
                    <input
                      type="checkbox"
                      checked={selected}
                      onChange={() => toggleOne(r.id)}
                      onClick={(e) => e.stopPropagation()}
                      style={{ cursor: "pointer" }}
                    />
                  </td>
                  <td style={styles.td}>{formatDate(r.date)}</td>
                  <td style={styles.td}>{VENUE_MAP[r.venue] || r.venue}</td>
                  <td style={{ ...styles.td, textAlign: "center" }}>{r.race_num}</td>
                  <td style={styles.td}>{TRACK_MAP[r.track] || r.track}</td>
                  <td style={{ ...styles.td, textAlign: "right" }}>{r.distance}</td>
                  <td style={styles.td}>{GRADE_MAP[r.grade] || r.grade || ""}</td>
                  <td style={{ ...styles.td, textAlign: "right" }}>{r.horse_count}</td>
                  <td style={styles.td}>
                    {r.race_name}
                    {r.areta && <span style={styles.badge}>荒れ</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
