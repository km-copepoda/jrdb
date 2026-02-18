import React, { useState, useCallback } from "react";

const MODEL_LABELS = {
  SED: "成績(SED)",
  KYI: "前日情報(KYI)",
  BAC: "レース条件(BAC)",
  KAB: "開催情報(KAB)",
};

export default function FieldSelector({
  availableFields,
  selectedFields,
  setSelectedFields,
}) {
  const [openSections, setOpenSections] = useState(() => {
    const init = {};
    for (const model of Object.keys(availableFields)) {
      init[model] = true;
    }
    return init;
  });

  const toggle = useCallback((model) => {
    setOpenSections((prev) => ({ ...prev, [model]: !prev[model] }));
  }, []);

  const handleCheck = useCallback(
    (key) => {
      setSelectedFields((prev) => {
        const next = new Set(prev);
        if (next.has(key)) {
          next.delete(key);
        } else {
          next.add(key);
        }
        return next;
      });
    },
    [setSelectedFields]
  );

  const selectAll = useCallback(
    (model) => {
      setSelectedFields((prev) => {
        const next = new Set(prev);
        for (const [key] of availableFields[model]) {
          next.add(`${model}.${key}`);
        }
        return next;
      });
    },
    [availableFields, setSelectedFields]
  );

  const deselectAll = useCallback(
    (model) => {
      setSelectedFields((prev) => {
        const next = new Set(prev);
        for (const [key] of availableFields[model]) {
          next.delete(`${model}.${key}`);
        }
        return next;
      });
    },
    [availableFields, setSelectedFields]
  );

  return (
    <div style={{ fontFamily: "sans-serif", fontSize: 14 }}>
      <div
        style={{
          padding: "8px 12px",
          marginBottom: 12,
          background: "#f0f4ff",
          borderRadius: 6,
          fontWeight: "bold",
        }}
      >
        選択フィールド: {selectedFields.size}件
      </div>

      {Object.keys(availableFields).map((model) => {
        const fields = availableFields[model];
        const isOpen = openSections[model];

        return (
          <div
            key={model}
            style={{
              border: "1px solid #ccc",
              borderRadius: 6,
              marginBottom: 8,
              overflow: "hidden",
            }}
          >
            {/* Header */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "8px 12px",
                background: "#f7f7f7",
                cursor: "pointer",
                userSelect: "none",
              }}
              onClick={() => toggle(model)}
            >
              <span style={{ fontWeight: "bold" }}>
                {isOpen ? "▼" : "▶"}{" "}
                {MODEL_LABELS[model] || model}
              </span>

              <span
                onClick={(e) => e.stopPropagation()}
                style={{ display: "flex", gap: 6 }}
              >
                <button
                  type="button"
                  onClick={() => deselectAll(model)}
                  style={{
                    fontSize: 12,
                    padding: "2px 8px",
                    cursor: "pointer",
                    border: "1px solid #aaa",
                    borderRadius: 4,
                    background: "#fff",
                  }}
                >
                  選択全解除
                </button>
                <button
                  type="button"
                  onClick={() => selectAll(model)}
                  style={{
                    fontSize: 12,
                    padding: "2px 8px",
                    cursor: "pointer",
                    border: "1px solid #aaa",
                    borderRadius: 4,
                    background: "#fff",
                  }}
                >
                  全選択
                </button>
              </span>
            </div>

            {/* Fields */}
            {isOpen && (
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: "4px 16px",
                  padding: "8px 12px",
                }}
              >
                {fields.map(([key, label]) => {
                  const fullKey = `${model}.${key}`;
                  return (
                    <label
                      key={fullKey}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 4,
                        cursor: "pointer",
                        whiteSpace: "nowrap",
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={selectedFields.has(fullKey)}
                        onChange={() => handleCheck(fullKey)}
                      />
                      {label}
                    </label>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
