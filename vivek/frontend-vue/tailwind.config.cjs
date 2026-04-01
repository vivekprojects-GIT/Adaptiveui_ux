/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{vue,ts,tsx,js,jsx}'],
  theme: {
    extend: {
      colors: {
        slate: {
          900: '#0F1419',
          800: '#1E293B',
        },
        cyan: {
          500: '#06B6D4',
        },
        emerald: {
          500: '#10B981',
        },
        amber: {
          500: '#F59E0B',
        },
        red: {
          500: '#EF4444',
        },
        purple: {
          500: '#8B5CF6',
        },
      },
      backgroundColor: {
        glass: 'rgba(15, 20, 25, 0.4)',
        'glass-lg': 'rgba(15, 20, 25, 0.6)',
      },
      borderColor: {
        glass: 'rgba(255, 255, 255, 0.1)',
        'glass-lg': 'rgba(255, 255, 255, 0.15)',
      },
    },
  },
  plugins: [],
}

