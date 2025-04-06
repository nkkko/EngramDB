/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./**/*.{js,jsx,ts,tsx,md,mdx}",
    "./node_modules/@mintlify/components/**/*.{js,jsx,ts,tsx}",
    "./.mintlify/**/*.{js,jsx,ts,tsx,md,mdx}"
  ],
  theme: {
    extend: {
      typography: {
        DEFAULT: {
          css: {
            maxWidth: 'none',
          },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
