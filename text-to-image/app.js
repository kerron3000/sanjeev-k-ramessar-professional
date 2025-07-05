const API_TOKEN = 'YOUR_HF_TOKEN_HERE';
const MODEL = 'black-forest-labs/FLUX.1-dev';

document.getElementById('generate').onclick = async () => {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) {
    alert('Please enter a prompt.');
    return;
  }

  const btn = document.getElementById('generate');
  btn.disabled = true;
  btn.textContent = 'Generating...';

  try {
    const res = await fetch(`https://api-inference.huggingface.co/models/${MODEL}`, {
      headers: {
        Authorization: `Bearer ${API_TOKEN}`,
        Accept: 'application/json'
      },
      method: 'POST',
      body: JSON.stringify({
        inputs: prompt,
        parameters: {
          width: 1024,
          height: 1024,
          guidance_scale: 3.5,
          num_inference_steps: 50,
          seed: Math.floor(Math.random() * 1e9),
        }
      })
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const blob = await res.blob();
    const imgUrl = URL.createObjectURL(blob);
    document.getElementById('output-image').src = imgUrl;

  } catch (e) {
    console.error(e);
    alert('Error generating image: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate Image';
  }
};
