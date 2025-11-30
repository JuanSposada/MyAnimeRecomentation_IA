// static/index.js

const API_BASE_URL = 'http://127.0.0.1:5000/recommend'; 

async function getRecommendations() {
    const animeName = document.getElementById('animeInput').value.trim();
    const resultsArea = document.getElementById('resultsArea');
    const spinner = document.getElementById("spinner");

    // Limpiar pantalla
    resultsArea.innerHTML = "";

    if (!animeName) {
        resultsArea.innerHTML = '<p class="error">Por favor, ingresa el nombre de un anime.</p>';
        return;
    }

    // Mostrar spinner
    spinner.style.display = "block";

    const url = `${API_BASE_URL}?anime=${encodeURIComponent(animeName)}`;

    try {
        const response = await fetch(url);
        
        // Ocultar spinner
        spinner.style.display = "none";

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({
                message: "Error desconocido del servidor."
            }));
            resultsArea.innerHTML = `<p class="error">⚠️ Error (${response.status}): ${errorData.message}</p>`;
            return;
        }

        const data = await response.json();

        let htmlContent = `<h2>Animes recomendados similares a <i>${data.input_anime}</i>:</h2>`;

        if (data.recommendations && data.recommendations.length > 0) {
            htmlContent += `<div class="recommendation-grid">`;

            data.recommendations.forEach((item, index) => {
                htmlContent += `
                    <div class="anime-card">
                        <img src="${item.image_url}" alt="Poster de ${item.name}" loading="lazy">
                        <div class="card-info">
                            <strong>${index + 1}. ${item.name}</strong>
                            <span class="score-pill">Similitud: ${item.similarity_score}</span>
                        </div>
                    </div>
                `;
            });

            htmlContent += `</div>`;
        } else {
            htmlContent += `<p>No se encontraron recomendaciones.</p>`;
        }

        resultsArea.innerHTML = htmlContent;

    } catch (error) {
        console.error("Error al conectar con la API:", error);
        spinner.style.display = "none"; 
        resultsArea.innerHTML = `<p class="error">❌ No se pudo conectar con el servidor Flask.</p>`;
    }
}
