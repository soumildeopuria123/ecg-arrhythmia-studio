/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        ink: '#0A0A0A',
        panel: '#141414',
        border: '#303030',
        signal: '#00F0FF',
        lime: '#00FF66',
        crimson: '#FF0055',
        yellow: '#FFE600',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Geist Mono', 'monospace'],
      },
      boxShadow: {
        hard: '4px 4px 0px 0px #FFFFFF',
      },
    },
  },
  plugins: [],
}