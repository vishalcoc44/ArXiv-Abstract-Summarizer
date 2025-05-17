console.log("script_rag.js: Global - Script loaded.");

document.addEventListener('DOMContentLoaded', () => {
    console.log("script_rag.js: DOMContentLoaded - Event fired.");

    const userInput = document.getElementById('userInput');
    const sendButton = document.querySelector('.send-button');
    const chatAreaContent = document.querySelector('.chat-area');
    const initialWelcomeMessage = document.querySelector('.welcome-message');
    const localModelButton = document.querySelector('.local-model-button');
    const searchButton = document.querySelector('.search-button');

    if (!userInput) console.error("script_rag.js: CRITICAL - userInput element NOT FOUND!");
    if (!sendButton) console.error("script_rag.js: CRITICAL - sendButton element NOT FOUND!");
    if (!chatAreaContent) console.error("script_rag.js: CRITICAL - chatAreaContent element NOT FOUND!");
    if (!localModelButton) console.log("script_rag.js: NOTE - localModelButton not found, will use default model");
    if (!searchButton) console.log("script_rag.js: NOTE - searchButton not found, will use default model");

    // Default to local model
    let currentModelType = 'local';
    let localModelLoaded = false; // Track if local model is loaded
    let modelCompiled = false; // Track if model is compiled for optimal performance
    let modelDevice = 'none'; // Track model device placement
    
    console.log("script_rag.js: DOMContentLoaded - Initial settings:",
               "model:", currentModelType);
    
    // Check if model is loaded on page load
    checkModelStatus();
    
    // Periodically check model status with progressive backoff
    let statusCheckInterval = 2000; // Start with 2 seconds
    const maxStatusCheckInterval = 10000; // Max 10 seconds between checks
    let statusCheckTimer = setInterval(() => {
        checkModelStatus();
        // Increase interval time if model is still loading (progressive backoff)
        if (!localModelLoaded && statusCheckInterval < maxStatusCheckInterval) {
            clearInterval(statusCheckTimer);
            statusCheckInterval = Math.min(statusCheckInterval * 1.5, maxStatusCheckInterval);
            statusCheckTimer = setInterval(checkModelStatus, statusCheckInterval);
        }
    }, statusCheckInterval);
    
    // Request notification permission to alert user when model is ready
    if ("Notification" in window && Notification.permission !== "granted") {
        Notification.requestPermission();
    }
    
    function showToast(message, duration = 3000) {
        // Create toast if it doesn't exist
        let toast = document.getElementById('toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'toast';
            document.body.appendChild(toast);
        }
        
        // Set message and show toast
        toast.textContent = message;
        toast.classList.add('show');
        
        // Hide toast after duration
        setTimeout(() => {
            toast.classList.remove('show');
        }, duration);
    }
    
    function checkModelStatus() {
        fetch('/api/model-status')
            .then(response => response.json())
            .then(data => {
                const wasLoaded = localModelLoaded;
                localModelLoaded = data.loaded;
                modelCompiled = data.compiled;
                modelDevice = data.device;
                
                console.log("script_rag.js: Model status check - loaded:", localModelLoaded, 
                           "compiled:", modelCompiled, "device:", modelDevice);
                
                // Update UI based on model status
                if (localModelLoaded) {
                    if (localModelButton) {
                        localModelButton.classList.remove('loading');
                        // Show if model is optimized (compiled)
                        const optimizedIndicator = modelCompiled ? '⚡ ' : '';
                        const deviceIndicator = modelDevice.includes('cuda') ? '🚀 ' : '';
                        localModelButton.innerHTML = `<i class="fas fa-microchip"></i> ${optimizedIndicator}${deviceIndicator}Local Model`;
                        
                        // Notify user when model becomes available
                        if (!wasLoaded && localModelLoaded) {
                            if (Notification.permission === "granted") {
                                new Notification("Model Ready", {
                                    body: "The local AI model is now loaded and ready to use!",
                                    icon: "/static_rag/favicon.ico"
                                });
                            }
                            appendMessage("Local model is now loaded and ready! " + 
                                         (modelCompiled ? "Running in optimized mode. " : "") +
                                         (modelDevice.includes('cuda') ? "Using GPU acceleration." : "Running on CPU."), 
                                         'system-message');
                        }
                    }
                    
                    // Once model is loaded and stable, check less frequently
                    if (statusCheckTimer && statusCheckInterval < maxStatusCheckInterval) {
                        clearInterval(statusCheckTimer);
                        statusCheckTimer = setInterval(checkModelStatus, maxStatusCheckInterval);
                    }
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
            console.log("script_rag.js: Switched to local model. 'currentModelType' is now:", currentModelType);
        });
    }

    if (searchButton) {
        searchButton.addEventListener('click', () => {
            currentModelType = 'gemini';
            searchButton.classList.add('active-model');
            if (localModelButton) localModelButton.classList.remove('active-model');
            if (userInput) userInput.placeholder = "Ask Gemini model...";
            console.log("script_rag.js: Switched to Gemini model. 'currentModelType' is now:", currentModelType);
        });
    }

    // Function to call Chat API with appropriate model
    async function performChatRequest(query, modelType) {
        console.log(`script_rag.js: ==> EXECUTING performChatRequest() with query: "${query}", model: "${modelType}"`);
        
        // Disable send button and show processing animation
        if (sendButton) {
            sendButton.classList.add('processing');
            sendButton.disabled = true;
        }

        // If local model is selected but not loaded, use Gemini instead
        if (modelType === 'local' && !localModelLoaded) {
            appendMessage("Local model is still loading. Using Gemini API instead.", 'system-message');
            modelType = 'gemini';
        }
        
        const modelName = modelType === 'gemini' ? 'Gemini' : 'Local';
        
        // Only show loading time warning for local model (with device-specific info)
        let loadingMessage = `Processing query with ${modelName} model...`;
        if (modelType === 'local') {
            const deviceInfo = modelDevice.includes('cuda') ? "GPU" : "CPU";
            const compiledInfo = modelCompiled ? "optimized" : "standard";
            const timeEstimate = modelDevice.includes('cuda') ? "2-5" : "5-15";
            
            loadingMessage += `\n\nRunning on ${deviceInfo} in ${compiledInfo} mode. May take ${timeEstimate} seconds.`;
        }
        appendMessage(loadingMessage, 'system-message');
        
        // Show typing indicator
        const typingIndicator = document.createElement('div');
        typingIndicator.classList.add('message', 'bot-message', 'typing-indicator');
        typingIndicator.innerHTML = '<span></span><span></span><span></span>';
        chatAreaContent.appendChild(typingIndicator);
        
        // Record start time for performance tracking
        const startTime = performance.now();
        
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
            
            const endTime = performance.now();
            const generationTime = ((endTime - startTime) / 1000).toFixed(2);
            
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
                const modelUsed = data.model_type === 'gemini' ? 'Gemini' : 'Local Model';
                const mainResponseText = data.response;
                const metaText = `\n<small class='response-meta'>(${modelUsed} - ${generationTime}s)</small>`;

                const messageContentElement = appendMessage('', 'bot-message'); 
                
                // Stream the main response text and get the raw content
                const rawStreamedText = await streamText(messageContentElement, mainResponseText, 1); // ULTRA FAST // charDelay for words, adjust as needed
                
                // Parse streamed text with marked.js if available
                if (typeof marked !== 'undefined') {
                    messageContentElement.innerHTML = marked.parse(rawStreamedText);
                } else {
                    // Fallback if marked.js isn't loaded - should not happen with CDN
                    messageContentElement.textContent = rawStreamedText; 
                }

                // Append meta information after Markdown parsing
                const metaElement = document.createElement('div'); // Use div for proper block display of meta
                metaElement.innerHTML = metaText.replace(/^\n+/, '').replace(/\n/g, '<br>'); // Clean leading newlines for meta
                messageContentElement.appendChild(metaElement);

                // Ensure final scroll after all content is added
                const chatAreaUsesColumnReverse = getComputedStyle(chatAreaContent).flexDirection === 'column-reverse';
                if (chatAreaUsesColumnReverse) {
                    chatAreaContent.scrollTop = 0;
                } else {
                    chatAreaContent.scrollTop = chatAreaContent.scrollHeight;
                }

                // Show warning if local model failed and Gemini was used as fallback
                if (data.warning) {
                    appendMessage(data.warning, 'system-message error-message');
                    if (data.error_details) {
                        appendMessage(`Details: ${data.error_details}`, 'system-message error-message');
                    }
                }
            } else {
                throw new Error(`Invalid response from ${modelName} model API`);
            }
        } catch (error) {
            // Remove typing indicator if still present
            if (typingIndicator.parentNode) {
                typingIndicator.parentNode.removeChild(typingIndicator);
            }
            
            console.error(`script_rag.js: performChatRequest Error (${modelType}):`, error);
            appendMessage(`Error with ${modelName} model: ${error.message}`, 'error-message');
        } finally {
            // Re-enable send button and remove processing animation
            if (sendButton) {
                sendButton.classList.remove('processing');
                sendButton.disabled = false;
            }
        }
    }

    // New function to stream text with animation
    async function streamText(element, text, charDelay = 20) { // charDelay in ms
        element.innerHTML = ''; // Clear previous content
        // const chars = Array.from(text); // For character-by-character
        const words = text.split(/(\s+|\n+)/); // Split by spaces or newlines, keeping delimiters
        let rawTextContent = '';

        for (let i = 0; i < words.length; i++) {
            const word = words[i];
            if (word === '\n' || word.match(/^\n+$/)) { // Handle one or more newlines
                element.appendChild(document.createElement('br'));
                rawTextContent += '\n';
            } else if (word === ' ' || word.match(/^\s+$/) && !word.includes('\n')) { // Handle one or more spaces (not newlines)
                element.appendChild(document.createTextNode(word)); 
                rawTextContent += word;
            } else if (word.trim() !== '') { // Non-empty, non-space, non-newline words
                const wordSpan = document.createElement('span');
                wordSpan.className = 'streamed-char'; // Reuse for similar animation, or create .streamed-word
                wordSpan.textContent = word;
                element.appendChild(wordSpan);
                rawTextContent += word;
                // Optional: Adjust delay for words if charDelay feels too slow for words
                await new Promise(resolve => setTimeout(resolve, charDelay)); 
            }
            
            const chatAreaUsesColumnReverse = getComputedStyle(chatAreaContent).flexDirection === 'column-reverse';
            if (chatAreaUsesColumnReverse) {
                chatAreaContent.scrollTop = 0;
            } else {
                chatAreaContent.scrollTop = chatAreaContent.scrollHeight;
            }
        }
        return rawTextContent; // Return the raw text for Markdown parsing
    }

    // Listener for the Send Button
    if (sendButton && userInput) {
        sendButton.addEventListener('click', async () => {
            const messageText = userInput.value.trim();
            
            // Log current model information
            console.log("----------------------------------------------------");
            console.log("script_rag.js: SendButtonClick - 'currentModelType':", currentModelType);
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
                console.log("script_rag.js: SendButtonClick - Message text is empty. Nothing to send.");
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
        ensureChatAreaVisible();
        
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', type);

        const messageContentElement = document.createElement('div');
        messageContentElement.className = 'message-content';
        
        if (type !== 'bot-message') { // For non-bot messages, set text directly
            const KINDA_SAFE_HTML = /<(\/?)?(b|strong|i|em|u|s|br|p|small|code|pre|ul|ol|li|div|span)(\s.*?)?>/gi;
            messageContentElement.innerHTML = text.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, "")
                                     .replace(KINDA_SAFE_HTML, (match) => match);
        } else if (text) { // For bot messages with initial text (e.g. error, system message before stream)
             if (typeof marked !== 'undefined') {
                messageContentElement.innerHTML = marked.parse(text);
            } else {
                messageContentElement.textContent = text;
            }
        }
        // If type is 'bot-message' and text is empty, it means we are preparing for streaming.

        messageDiv.appendChild(messageContentElement);

        // Add copy button for bot messages that are not system/error messages within bot context
        if (type === 'bot-message' && !messageDiv.classList.contains('system-message') && !messageDiv.classList.contains('error-message')) {
            const copyButton = document.createElement('button');
            copyButton.className = 'copy-to-clipboard-button';
            copyButton.innerHTML = '<i class="far fa-copy"></i>'; // Font Awesome icon
            copyButton.title = 'Copy to clipboard';
            messageDiv.appendChild(copyButton); // Append to messageDiv, not messageContentElement

            copyButton.addEventListener('click', () => {
                const textToCopy = messageContentElement.innerText || messageContentElement.textContent;
                navigator.clipboard.writeText(textToCopy).then(() => {
                    copyButton.innerHTML = '<i class="fas fa-check"></i>'; // Change to checkmark
                    setTimeout(() => {
                        copyButton.innerHTML = '<i class="far fa-copy"></i>'; // Revert to copy icon
                    }, 2000);
                }).catch(err => {
                    console.error('Failed to copy text: ', err);
                    // Optionally show an error toast or message
                    showToast("Failed to copy!", 2000);
                });
            });
        }

        const chatAreaUsesColumnReverse = getComputedStyle(chatAreaContent).flexDirection === 'column-reverse';
        if (chatAreaUsesColumnReverse) {
            chatAreaContent.insertBefore(messageDiv, chatAreaContent.firstChild);
        } else {
            chatAreaContent.appendChild(messageDiv);
        }
        
        if (chatAreaUsesColumnReverse) {
            chatAreaContent.scrollTop = 0;
        } else {
            chatAreaContent.scrollTop = chatAreaContent.scrollHeight;
        }

        console.log(`script_rag.js: Appended message - Type: ${type}, Text: ${text ? text.substring(0, 50) : '[empty for streaming]'}...`);
        return messageContentElement; // Return the element where content should be streamed/placed
    }

    // Handle initial response rendering if it was passed from server-side (e.g., after a POST with Flask render_template)
    // This is generally not the preferred way for SPAs, but might exist from previous versions.
    const dynamicResponseArea = document.getElementById('dynamic-response-area');
    if (dynamicResponseArea && dynamicResponseArea.textContent.trim() !== "" && dynamicResponseArea.textContent.trim() !== "None") {
        console.log("script_rag.js: Found initial server-rendered response. Appending to chat.");
        ensureChatAreaVisible();
        // Assume initial server response is from the bot
        appendMessage(dynamicResponseArea.textContent.trim(), 'bot-message');
        dynamicResponseArea.textContent = ""; // Clear it after processing
        dynamicResponseArea.style.display = 'none'; 
    }

    // Automatic focus on input field
    if (userInput) {
        userInput.focus();
    }

    // Initial UI update based on model selection buttons
    if (localModelButton && localModelButton.classList.contains('active-model')) {
        currentModelType = 'local';
        if (userInput) userInput.placeholder = "Ask local model...";
    } else if (searchButton && searchButton.classList.contains('active-model')) {
        currentModelType = 'gemini';
        if (userInput) userInput.placeholder = "Ask Gemini model...";
    } else if (localModelButton) { // Default to local if no active one is set explicitly
        localModelButton.classList.add('active-model');
        currentModelType = 'local';
        if (userInput) userInput.placeholder = "Ask local model...";
    } else if (searchButton) { // Fallback to search if local button doesn't exist
        searchButton.classList.add('active-model');
        currentModelType = 'gemini';
        if (userInput) userInput.placeholder = "Ask Gemini model...";
    }

    console.log("script_rag.js: DOMContentLoaded - End of setup.");
}); 