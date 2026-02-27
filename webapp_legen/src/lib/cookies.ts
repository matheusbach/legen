export function setCookie(name: string, value: unknown, days = 180) {
  const encoded = encodeURIComponent(JSON.stringify(value))
  const expires = new Date(Date.now() + days * 24 * 60 * 60 * 1000).toUTCString()
  document.cookie = `${name}=${encoded}; expires=${expires}; path=/`
}

export function getCookie<T>(name: string): T | null {
  const match = document.cookie.match(new RegExp(`(?:^|; )${name}=([^;]*)`))
  if (!match) return null
  try {
    return JSON.parse(decodeURIComponent(match[1])) as T
  } catch {
    return null
  }
}
