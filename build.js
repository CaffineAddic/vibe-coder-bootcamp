// Bundles everything into one self-contained HTML file: dist/vibe-coder-bootcamp.html
// Usage: node build.js
const fs = require("fs");
const { torchDataJs } = require("./tools/torch-files.js");
const read = f => fs.readFileSync(f, "utf8");
fs.writeFileSync("course/torchlite.js", torchDataJs("."));
let html = read("index.html");
html = html.replace('<link rel="stylesheet" href="style.css">', () => "<style>\n" + read("style.css") + "</style>");
html = html.replace(/<script src="((?:course\/)?[\w-]+\.js)"><\/script>/g, (_, f) => "<script>\n" + read(f) + "\n</script>");
fs.mkdirSync("dist", { recursive: true });
fs.writeFileSync("dist/vibe-coder-bootcamp.html", html);
// Artifact flavor: contents only, without the document wrapper
const inner = html.replace(/^[\s\S]*?<head>/, "").replace(/<\/head>\s*<body>/, "").replace(/<\/body>\s*<\/html>\s*$/, "")
  .replace(/<meta charset[^>]*>\s*/, "").replace(/<meta name="viewport"[^>]*>\s*/, "");
fs.writeFileSync("dist/artifact.html", inner);
console.log("Built dist/vibe-coder-bootcamp.html (" + html.length + " bytes)");
