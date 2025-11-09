'use strict';
// NB: THIS IS NOT MY ACTUAL website's Javascript file, you can add your own
// Chatbot functionality
const chatbotToggle = document.getElementById('chatbot-toggle');
const chatbotWindow = document.getElementById('chatbot-window');
const chatbotClose = document.getElementById('chatbot-close');
const chatbotMessages = document.getElementById('chatbot-messages');
const chatbotInput = document.getElementById('chatbot-input-field');
const chatbotSend = document.getElementById('chatbot-send');

// Toggle chatbot window
chatbotToggle.addEventListener('click', () => {
    chatbotWindow.style.display = chatbotWindow.style.display === 'flex' ? 'none' : 'flex';
});

chatbotClose.addEventListener('click', () => {
    chatbotWindow.style.display = 'none';
});

// Send message function
chatbotSend.addEventListener('click', sendMessage);
chatbotInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    const messageText = chatbotInput.value.trim();
    if (!messageText) return;

    // Add user message to chat
    const userMessage = document.createElement('div');
    userMessage.classList.add('chat-message', 'user');
    userMessage.textContent = messageText;
    chatbotMessages.appendChild(userMessage);

    // Clear input
    chatbotInput.value = '';

    // Scroll to bottom
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;

    // Show typing indicator
    const typingIndicator = document.createElement('div');
    typingIndicator.classList.add('chat-message', 'ai', 'typing');
    typingIndicator.textContent = 'Thinking...';
    chatbotMessages.appendChild(typingIndicator);
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;

    // Send query to FastAPI backend
    fetch('https://your-cloud-run-url.run.app/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question: messageText,
            format_type: 'html',
        }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Remove typing indicator
        typingIndicator.remove();

        // Add AI response
        const aiResponse = document.createElement('div');
        aiResponse.classList.add('chat-message', 'ai');
        aiResponse.innerHTML = data.answer;
        chatbotMessages.appendChild(aiResponse);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    })
    .catch(error => {
        // Remove typing indicator
        typingIndicator.remove();

        // Show error message
        const aiResponse = document.createElement('div');
        aiResponse.classList.add('chat-message', 'ai');
        aiResponse.textContent = 'Sorry, there was an error. Please try again later.';
        chatbotMessages.appendChild(aiResponse);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
        console.error('Error:', error);
    });
}
