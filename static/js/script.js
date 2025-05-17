console.log("script.js: Global - Script loaded.");

document.addEventListener('DOMContentLoaded', () => {
    console.log("script.js: DOMContentLoaded - Event fired.");

    const userInput = document.getElementById('userInput');
    const sendButton = document.querySelector('.send-button');
    const chatAreaContent = document.querySelector('.chat-area');
    const initialWelcomeMessage = document.querySelector('.welcome-message');
    const localModelButton = document.querySelector('.local-model-button');
    const searchButton = document.querySelector('.search-button');

    if (!userInput) console.error("script.js: CRITICAL - userInput element NOT FOUND!");
    if (!sendButton) console.error("script.js: CRITICAL - sendButton element NOT FOUND!");
    if (!chatAreaContent) console.error("script.js: CRITICAL - chatAreaContent element NOT FOUND!");
    if (!localModelButton) console.log("script.js: NOTE - localModelButton not found, will use default model");
    if (!searchButton) console.log("script.js: NOTE - searchButton not found, will use default model");

    // Default to local model
    let currentModelType = 'local';
    let localModelLoaded = false; // Track if local model is loaded
    console.log("script.js: DOMContentLoaded - Initial 'currentModelType' value:", currentModelType);
    
    // Check if model is loaded on page load
    checkModelStatus();
    
    // Periodically check model status
    const statusCheckInterval = setInterval(checkModelStatus, 5000);
    
    function checkModelStatus() {
        fetch('/api/model-status')
            .then(response => response.json())
            .then(data => {
                localModelLoaded = data.loaded;
                console.log("script.js: Model loaded status:", localModelLoaded);
                
                // Update UI based on model status
                if (localModelLoaded) {
                    if (localModelButton) {
                        localModelButton.classList.remove('loading');
                        localModelButton.innerHTML = '<i class="fas fa-microchip"></i> Local Model';
                    }
                    // Stop checking once loaded
                    clearInterval(statusCheckInterval);
                } else {
                    if (localModelButton) {
                        localModelButton.classList.add('loading');
                        localModelButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
                    }
                }
            })
            .catch(error => {
                console.error("Error checking model status:", error);
            });
    }

    function ensureChatAreaVisible() {
        if (initialWelcomeMessage && initialWelcomeMessage.parentNode === chatAreaContent) {
            chatAreaContent.removeChild(initialWelcomeMessage);
            if (chatAreaContent) chatAreaContent.style.justifyContent = 'flex-start';
        }
    }

    // Model selection listeners
    if (localModelButton) {
        localModelButton.addEventListener('click', () => {
            if (!localModelLoaded) {
                appendMessage("Local model is still loading. Using Gemini API until it's ready.", 'system-message');
                currentModelType = 'gemini';
                searchButton.classList.add('active-model');
                localModelButton.classList.remove('active-model');
                return;
            }
            
            currentModelType = 'local';
            localModelButton.classList.add('active-model');
            if (searchButton) searchButton.classList.remove('active-model');
            if (userInput) userInput.placeholder = "Ask local model...";
            console.log("script.js: Switched to local model. 'currentModelType' is now:", currentModelType);
        });
    }

    if (searchButton) {
        searchButton.addEventListener('click', () => {
            currentModelType = 'gemini';
            searchButton.classList.add('active-model');
            if (localModelButton) localModelButton.classList.remove('active-model');
            if (userInput) userInput.placeholder = "Ask Gemini model...";
            console.log("script.js: Switched to Gemini model. 'currentModelType' is now:", currentModelType);
        });
    }

    // Function to call Chat API with appropriate model
    async function performChatRequest(query, modelType) {
        console.log(`script.js: ==> EXECUTING performChatRequest() with query: "${query}", model: "${modelType}"`);
        
        // If local model is selected but not loaded, use Gemini instead
        if (modelType === 'local' && !localModelLoaded) {
            appendMessage("Local model is still loading. Using Gemini API instead.", 'system-message');
            modelType = 'gemini';
        }
        
        const modelName = modelType === 'gemini' ? 'Gemini' : 'Local';
        
        // Only show loading time warning for local model
        let loadingMessage = `Processing query with ${modelName} model: "${query}"...`;
        if (modelType === 'local') {
            loadingMessage += '\n\nLocal model processing may take 5-10 seconds per response.';
        }
        appendMessage(loadingMessage, 'system-message');
        
        // Show typing indicator
        const typingIndicator = document.createElement('div');
        typingIndicator.classList.add('message', 'bot-message', 'typing-indicator');
        typingIndicator.innerHTML = '<span></span><span></span><span></span>';
        chatAreaContent.appendChild(typingIndicator);
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    query: query,
                    model_type: modelType
                })
            });
            
            // Remove typing indicator
            if (typingIndicator.parentNode) {
                typingIndicator.parentNode.removeChild(typingIndicator);
            }
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(`API Error (${response.status}): ${errorData.error || response.statusText}`);
            }
            
            const data = await response.json();
            if (data && data.response) {
                appendMessage(data.response, 'bot-message');
            } else {
                throw new Error(`Invalid response from ${modelName} model API`);
            }
        } catch (error) {
            // Remove typing indicator if still present
            if (typingIndicator.parentNode) {
                typingIndicator.parentNode.removeChild(typingIndicator);
            }
            
            console.error(`script.js: performChatRequest Error (${modelType}):`, error);
            appendMessage(`Error with ${modelName} model: ${error.message}`, 'error-message');
        }
    }

    // Listener for the Send Button
    if (sendButton && userInput) {
        sendButton.addEventListener('click', async () => {
            const messageText = userInput.value.trim();
            
            // Log current model information
            console.log("----------------------------------------------------");
            console.log("script.js: SendButtonClick - 'currentModelType' at decision point:", currentModelType);
            console.log("----------------------------------------------------");

            if (messageText) {
                ensureChatAreaVisible();
                appendMessage(messageText, 'user-message');
                
                // If local model is selected but not loaded, use Gemini
                let effectiveModelType = currentModelType;
                if (currentModelType === 'local' && !localModelLoaded) {
                    effectiveModelType = 'gemini';
                }
                
                await performChatRequest(messageText, effectiveModelType);
                
                userInput.value = '';
                if (sendButton.classList.contains('active')) {
                    sendButton.classList.remove('active');
                }
                sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
                if (userInput) userInput.focus();
            } else {
                console.log("script.js: SendButtonClick - Message text is empty. Nothing to send.");
            }
        });

        // Other listeners (keypress, input)
        userInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                if (sendButton) sendButton.click();
            }
        });
        
        userInput.addEventListener('input', () => {
            if (userInput.value.trim() !== '') {
                if (!sendButton.classList.contains('active')) sendButton.classList.add('active');
                sendButton.innerHTML = '<i class="fas fa-arrow-up"></i>';
            } else {
                if (sendButton.classList.contains('active')) sendButton.classList.remove('active');
                sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
            }
        });
    }

    // Function to append messages to the chat area
    function appendMessage(text, type) {
        if (!chatAreaContent) {
            console.error("script.js: appendMessage - chatAreaContent is null!");
            return;
        }
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', type);
        let html = text;
        // Basic Markdown-like formatting (simplified for clarity, add more if needed)
        html = html.replace(/\n/g, '<br>');
        messageDiv.innerHTML = html;
        chatAreaContent.appendChild(messageDiv);
        chatAreaContent.scrollTop = chatAreaContent.scrollHeight;
    }

    // Initial UI setup
    if (initialWelcomeMessage && chatAreaContent && chatAreaContent.children.length === 1) {
        chatAreaContent.style.justifyContent = 'center';
    } else if (chatAreaContent) {
        chatAreaContent.style.justifyContent = 'flex-start';
    }
    
    // Set the initial active button if present
    if (localModelButton) {
        localModelButton.classList.add('active-model');
        if (!localModelLoaded) {
            localModelButton.classList.add('loading');
            localModelButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
        }
    }
    
    console.log("script.js: DOMContentLoaded - End of setup.");
}); 