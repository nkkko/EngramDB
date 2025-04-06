/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./**/*.{js,jsx,ts,tsx,md,mdx}",
    "./node_modules/@mintlify/components/**/*.{js,jsx,ts,tsx}",
    "./.mintlify/**/*.{js,jsx,ts,tsx,md,mdx}"
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#6366F1',
          light: '#818CF8',
          dark: '#4338CA',
        },
        secondary: {
          DEFAULT: '#8B5CF6',
          light: '#A78BFA',
          dark: '#6D28D9',
        },
      },
      typography: {
        DEFAULT: {
          css: {
            maxWidth: 'none',
            color: 'inherit',
            a: {
              color: '#6366F1',
              '&:hover': {
                color: '#4338CA',
              },
            },
            h1: {
              color: 'inherit',
            },
            h2: {
              color: 'inherit',
            },
            h3: {
              color: 'inherit',
            },
            h4: {
              color: 'inherit',
            },
            code: {
              color: 'inherit',
            },
            pre: {
              backgroundColor: '#1F2937',
            },
          },
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'gradient-primary': 'linear-gradient(to right, #6366F1, #8B5CF6)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.5s ease-in-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: 0 },
          '100%': { opacity: 1 },
        },
      },
      boxShadow: {
        glow: '0 0 15px 2px rgba(99, 102, 241, 0.3)',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
  mintlify: {
    icons: ['magnifying-glass', 'server', 'code', 'clock', 'python', 'browser', 'database', 'rocket', 'book', 'flask', 'time', 'link']
  }
}
