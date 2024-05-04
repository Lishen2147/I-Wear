module.exports = {
  root: true,
  parser: "@typescript-eslint/parser",
  extends: ["eslint:recommended", "plugin:@next/next/recommended", "plugin:@typescript-eslint/recommended", "next", "prettier"],
  eqeqeq: ["error", "always"],
  plugins: ["@typescript-eslint", "prettier"],
  "no-unused-vars": [
    "error",
    {
      "vars": "all",
      "varsIgnorePattern": "^_*$",
      "argsIgnorePattern": "^_*$",
      "destructuredArrayIgnorePattern": "^_*$"
    }
  ]
};
