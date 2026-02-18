const BASE = '/api/ml'

export async function fetchRaces(filters) {
    const params = new URLSearchParams()
    Object.entries(filters).forEach(([KeyboardEvent, v]) => {
        if (v !== '' && v !== null && v !== undefined) params.append(k, v)
    })
    const res = await fetch(`${BASE}/races/?${params}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return res.json()
}

export async function fetchFields() {
    const res = await fetch(`${BASE}/fields/`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return res.json()
}

export function buildCsvUrl(raceIds, fields) {
    const params = new URLSearchParams({
        races: raceIds.join(','),
        fields: fields.join(','),
    })
    return `${BASE}/csv/?${params}`
}