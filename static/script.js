const canvas = document.getElementById('canvas');
const recognizeButton = document.getElementById('recognize-button');
const clear_button = document.getElementById('clear-button');
const context = canvas.getContext('2d');
let isDrawing = false;

canvas.addEventListener('mousedown', (event) => {
  isDrawing = true;
  context.beginPath();
  context.moveTo(event.offsetX, event.offsetY);
});

canvas.addEventListener('mousemove', (event) => {
  if (isDrawing) {
    context.lineTo(event.offsetX, event.offsetY);
    context.strokeStyle = 'black';
    context.lineWidth = 30;
    context.lineCap = 'round';
    context.stroke();
  }
});

canvas.addEventListener('mouseup', () => {
  isDrawing = false;
});

recognizeButton.addEventListener('click', async () => {
  var image = canvas.toDataURL('image/png');

  const response = await fetch('/recognize/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({image:image})
  });

  const result = await response.json();
  document.querySelector('#result').textContent = result.value;
});

clear_button.addEventListener('click', () => {
  context.clearRect(0, 0, canvas.width, canvas.height);
  message.innerHTML='';
});


