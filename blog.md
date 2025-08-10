---
layout: default
title: Research Blog - Eitamar Saraf
permalink: /blog/
---

# Research Blog & Experiments

---

<div class="project-cards" style="display: flex; justify-content: center; margin-top: 2rem;">
  <div class="project-card" style="background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 2rem; width: 350px; min-height: 420px;">
    <h2 style="margin-top: 0; margin-bottom: 1rem;">🧩 Classic XOR Problem – Why Neural Networks Matter</h2>
    <div id="xor-plot" style="width:100%;height:300px;"></div>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script>
      const x = [0, 0, 1, 1];
      const y = [0, 1, 0, 1];
      const label = [0, 1, 1, 0];
      const colors = label.map(l => l === 1 ? 'red' : 'blue');
      const trace = {
        x: x,
        y: y,
        mode: 'markers',
        marker: {
          color: colors,
          size: 18,
          line: { width: 2, color: 'black' }
        },
        text: label.map(l => `Class: ${l}`),
        type: 'scatter'
      };
      const layout = {
        title: '',
        xaxis: { title: 'Input X1', range: [-0.5, 1.5] },
        yaxis: { title: 'Input X2', range: [-0.5, 1.5] },
        showlegend: false,
        margin: { t: 10 }
      };
      Plotly.newPlot('xor-plot', [trace], layout);
    </script>
    <p style="margin-top: 1.5rem; color: #555;">
      The XOR problem is a classic example that shows why simple models fail and neural networks succeed. Each color represents a class—can you separate them with a straight line?
    </p>
  </div>
</div>

---

