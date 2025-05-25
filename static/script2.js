document.addEventListener('DOMContentLoaded', () => {
    // Add dynamic loading for marked.js as a fallback
    function loadMarkedLibrary() {
        return new Promise((resolve, reject) => {
            // Check if marked is already defined (from the CDN)
            if (typeof marked !== 'undefined') {
                console.log('[DEBUG] Marked.js already loaded from CDN');
                resolve(marked);
                return;
            }

            console.log('[DEBUG] Attempting to load Marked.js dynamically');
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/marked/4.3.0/marked.min.js';
            script.onload = () => {
                console.log('[DEBUG] Marked.js loaded dynamically');
                resolve(marked);
            };
            script.onerror = () => {
                console.error('[DEBUG] Failed to load Marked.js dynamically');
                reject(new Error('Failed to load Marked.js'));
            };
            document.head.appendChild(script);
        });
    }

    // Helper function to safely parse markdown with fallback to plain text
    function safeMarkdownParse(text) {
        if (typeof marked !== 'undefined') {
            try {
                return marked.parse(text, { sanitize: false, mangle: false });
            } catch (e) {
                console.error('[DEBUG] Error in marked.parse():', e);
                return text; // Fallback to plain text
            }
        }
        // Basic markdown to HTML conversion as fallback
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>')             // Italic
            .replace(/`(.*?)`/g, '<code>$1</code>')           // Code
            .replace(/\n\n/g, '<br><br>')                     // Line breaks
            .replace(/\n\*(.*)/g, '<br>• $1');                // Bullet points
    }

    // Animate HTML content with a typewriter effect
    function animateTextByWord(container, htmlContent) {
        console.log('[DEBUG] Starting typewriter animation for container:', container);
        
        // Set the HTML content first
        container.innerHTML = htmlContent;

        const textNodes = [];
        const walk = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null, false);
        let node;
        while(node = walk.nextNode()) {
            if (node.nodeValue.trim() !== '') {
                textNodes.push(node);
            }
        }

        if (textNodes.length === 0) {
            gsap.fromTo(container, { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.3, ease: "power1.inOut" });
            return null; 
        }

        const originalContents = textNodes.map(n => ({ parent: n.parentNode, node: n, originalText: n.nodeValue }));
        originalContents.forEach(item => item.node.nodeValue = ''); 

        let charCount = 0;
        const tl = gsap.timeline();

        originalContents.forEach(item => {
            const chars = item.originalText.split('');
            const tempWrapper = document.createDocumentFragment(); 

            chars.forEach(char => {
                const span = document.createElement('span');
                span.textContent = char;
                span.style.opacity = '0'; 
                span.style.display = 'inline-block'; 
                tempWrapper.appendChild(span);
                
                tl.to(span, {
                    opacity: 1,
                    duration: 0.01, 
                }, charCount * 0.005); 
                charCount++;
            });
            if (item.parent) {
                 item.parent.insertBefore(tempWrapper, item.node); 
            }
        });

        return tl;
    }

    loadMarkedLibrary().catch(err => {
        console.warn('[DEBUG] Using basic markdown parser fallback:', err);
    });

    let activeThinkingAnimation = null; 
    const loadedLottieAnimations = new Map(); 

    function loadLottieAnimation(container, path, loop = true, autoplay = true, renderer = 'svg', customConfig = {}) {
        const animationID = customConfig.animationID; // May be undefined

        if (typeof lottie === 'undefined') {
            console.error('[LOTTIE CRITICAL] Lottie library (lottie.min.js) is not loaded. Cannot play animations.');
            if (container) container.innerHTML = "<div style='color:red; font-size:10px;'>Lottie Lib Err</div>";
            return null;
        }

        if (!container) {
            console.error('[LOTTIE DEBUG] Lottie container is null for path:', path);
            return null;
        }
        console.log(`[LOTTIE DEBUG] Attempting to load: ${path} into container. ID: ${animationID}`, container);

        // Destroy any existing animation in THIS specific container
        loadedLottieAnimations.forEach((animInstance, id_in_cache) => {
            if (animInstance.wrapper === container) {
                console.log(`[LOTTIE DEBUG] Destroying existing Lottie (ID: ${id_in_cache}, Path: ${animInstance.path}) in target container before new load.`);
                try {
                    animInstance.destroy();
                } catch (e) {
                    console.error(`[LOTTIE DEBUG] Error destroying animation ID ${id_in_cache}:`, e);
                }
                loadedLottieAnimations.delete(id_in_cache);
            }
        });
        container.innerHTML = ''; // Clear container AFTER destroying, before loading new

        // If an animationID is provided, and it happens to still be in the cache for some reason
        // (e.g. if destroy failed or logic error), remove it to prevent conflicts if we are re-loading with the same ID.
        if (animationID && loadedLottieAnimations.has(animationID)) {
            console.warn(`[LOTTIE DEBUG] Animation ID ${animationID} was still in cache. Removing before new load with same ID.`);
            const oldAnimForID = loadedLottieAnimations.get(animationID);
            if (oldAnimForID.wrapper !== container) {
                 // This case is tricky: same ID but different container. This shouldn't happen with good ID management.
                 console.warn(`[LOTTIE DEBUG] ID ${animationID} was for a different container. This is unusual.`);
            } else {
                // It was for the same container but wasn't cleaned by the loop above? Should not happen.
            }
            try { oldAnimForID.destroy(); } catch(e) { /*ignore*/ }
            loadedLottieAnimations.delete(animationID);
        }

        try {
            console.log(`[LOTTIE DEBUG] Calling lottie.loadAnimation for path: ${path}`);
            const animation = lottie.loadAnimation({
                container: container,
                renderer: renderer,
                loop: loop,
                autoplay: autoplay,
                path: path,
                ...customConfig // Spread customConfig which might contain animationID if we need it later
            });

            console.log('[LOTTIE DEBUG] lottie.loadAnimation call completed. Returned object:', animation);

            if (!animation || typeof animation.play !== 'function' || typeof animation.destroy !== 'function') {
                console.error('[LOTTIE DEBUG] Loaded animation object is invalid or incomplete for path:', path, 'Object:', animation);
                container.innerHTML = "<div style='color:red; font-size:10px;'>Anim Load Fail</div>";
                return null;
            }

            console.log(`[LOTTIE DEBUG] Animation for path ${path} seems valid.`);
            if (animationID) {
                console.log(`[LOTTIE DEBUG] Caching animation with ID ${animationID} for path: ${path}`);
                loadedLottieAnimations.set(animationID, animation);
            } else {
                console.log(`[LOTTIE DEBUG] Loaded animation (no ID provided for caching) for path: ${path}`);
            }
            return animation;
        } catch (error) {
            console.error(`[LOTTIE DEBUG] Exception during lottie.loadAnimation for path: ${path}`, error);
            container.innerHTML = "<div style='color:red; font-size:10px;'>Anim Excp</div>"; 
            return null;
        }
    }

    // Sidebar elements
    const sidebar = document.querySelector('.sidebar');
    const mainContentSidebarToggleBtn = document.querySelector('.chat-header .current-chat-name .icon-btn i.fa-chevron-left');
    const navSectionHeaders = document.querySelectorAll('.sidebar-nav .nav-section-header');
    const newChatButton = document.querySelector('.new-chat-btn');
    const addFolderButton = document.getElementById('add-folder-btn');
    const foldersListUL = document.getElementById('folders-list');
    const directChatsListUL = document.getElementById('direct-chats-list');

    // Modal elements
    const inputModal = document.getElementById('inputModal');
    const modalTitleEl = document.getElementById('modalTitle');
    const modalInputEl = document.getElementById('modalInput');
    const modalSubmitBtn = document.getElementById('modalSubmitBtn');
    const modalCancelBtn = document.getElementById('modalCancelBtn');

    // Folder Select Modal elements
    const folderSelectModal = document.getElementById('folderSelectModal');
    const folderSelectModalTitle = document.getElementById('folderSelectModalTitle');
    const folderSelectList = document.getElementById('folderSelectList');
    const folderSelectCancelBtn = document.getElementById('folderSelectCancelBtn');

    // Action Choice Modal elements
    const actionChoiceModal = document.getElementById('actionChoiceModal');
    const actionChoiceModalTitle = document.getElementById('actionChoiceModalTitle');
    const actionChoiceModalMessage = document.getElementById('actionChoiceModalMessage');
    const actionChoiceOption1Btn = document.getElementById('actionChoiceOption1Btn');
    const actionChoiceOption2Btn = document.getElementById('actionChoiceOption2Btn');
    const actionChoiceOption3Btn = document.getElementById('actionChoiceOption3Btn');
    const actionChoiceCancelBtn = document.getElementById('actionChoiceCancelBtn');

    // Main content elements
    const chatViewContainer = document.getElementById('chat-view');
    const initialPromptArea = document.querySelector('.initial-prompt-area');
    const chatOutput = document.getElementById('chat-output');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const mainContentTitle = document.querySelector('.initial-prompt-area h1'); 
    const suggestionCards = document.querySelectorAll('.suggestion-cards .card');
    const chatHeaderTitleElement = document.querySelector('.chat-header .current-chat-name h2');
    const exportChatBtn = document.getElementById('export-chat-btn'); // Added export button
    const clearAllChatsBtn = document.getElementById('clear-all-chats-btn'); // Added clear all chats button

    let currentActiveChatId = null; // To store the ID of the currently active chat

    // --- GSAP Initial Load Animations ---
    const pageLoadTl = gsap.timeline({ defaults: { ease: 'expo.out', duration: 0.7 } });
    pageLoadTl
        .from('.sidebar-header', { opacity: 0, x: -40, duration: 0.6, delay: 0.2 })
        .from('.search-bar', { opacity: 0, x: -40, duration: 0.5 }, "-=0.4")
        .from('.new-chat-btn', { opacity: 0, y: 30, duration: 0.5 }, "-=0.2");
    pageLoadTl
        .from('.chat-header', { opacity: 0, y: -30, duration: 0.6 }, "-=0.8") 
        .add(() => {
            if (initialPromptArea && getComputedStyle(initialPromptArea).display !== 'none') {
                gsap.from(initialPromptArea.querySelector('.main-prompt-logo'), { opacity:0, scale: 0.3, duration: 0.7, ease: 'elastic.out(1, 0.7)', delay: 0.1}); 
                if (mainContentTitle) splitAndAnimateH1(mainContentTitle, 0.2);
                gsap.from(initialPromptArea.querySelector('.prompt-subtitle'), { opacity:0, y: 20, duration: 0.6, ease: 'power2.out', delay: 0.4 });
                gsap.from(suggestionCards, { opacity: 0, y: 25, duration: 0.5, stagger: 0.08, ease: 'power2.out', delay: 0.6 });
                gsap.from('.content-type-filter button', { opacity: 0, y: 20, duration: 0.4, stagger: 0.04, ease: 'power2.out', delay: 0.9 });
            }
        }, "-=0.5") 
        .from('.chat-input-footer', { opacity: 0, y: 30, duration: 0.6 }, "-=0.3"); 
        
    const existingMessages = chatOutput.querySelectorAll('.message');
    if (existingMessages.length > 0 && initialPromptArea) {
        initialPromptArea.style.display = 'none'; 
        gsap.set(chatOutput, {opacity: 1}); 
    }

    // --- Interactive Element Animations ---
    function applyButtonAnimations(selector) {
        const buttons = document.querySelectorAll(selector);
        buttons.forEach(button => {
            // Hover animation
            button.addEventListener('mouseenter', () => {
                gsap.to(button, { 
                    scale: 1.05, 
                    filter: 'brightness(1.1)',
                    duration: 0.2, 
                    ease: 'power1.out' 
                });
            });
            button.addEventListener('mouseleave', () => {
                gsap.to(button, { 
                    scale: 1, 
                    filter: 'brightness(1)',
                    duration: 0.2, 
                    ease: 'power1.inOut' 
                });
            });

            // Click animation
            button.addEventListener('mousedown', () => {
                gsap.to(button, { 
                    scale: 0.95, 
                    filter: 'brightness(0.9)',
                    duration: 0.1, 
                    ease: 'power1.out' 
                });
            });
            button.addEventListener('mouseup', () => { // Also handle mouseup outside
                gsap.to(button, { 
                    scale: 1.05, // Return to hover state if mouse is still over
                    filter: 'brightness(1.1)',
                    duration: 0.1, 
                    ease: 'power1.inOut' 
                });
            });
             button.addEventListener('click', () => { // Ensure it returns to a good state after click
                gsap.to(button, { 
                    scale: 1.05, // Match hover state briefly
                    filter: 'brightness(1.1)', 
                    duration: 0.05, 
                    onComplete: () => {
                        // If mouse is not over after click, return to normal
                        if (!button.matches(':hover')) {
                             gsap.to(button, { scale: 1, filter: 'brightness(1)', duration: 0.1});
                        }
                    }
                });
            });
        });
    }

    // Apply to general icon buttons
    applyButtonAnimations('.icon-btn');
    applyButtonAnimations('.new-chat-btn');
    applyButtonAnimations('.model-btn');
    applyButtonAnimations('.filter-btn');
    // Add other selectors if needed, e.g., modal buttons:
    // applyButtonAnimations('.modal-btn');


    // --- GSAP Helper Functions (e.g., Typewriter, H1 split) ---
    function splitAndAnimateH1(element, delay = 0) {
        if (!element) return;
        const text = element.textContent;
        element.innerHTML = ''; 
        text.split('').forEach(char => {
            const span = document.createElement('span');
            span.className = 'char';
            span.textContent = char === ' ' ? '\u00A0' : char; 
            element.appendChild(span);
        });
        gsap.to(element.querySelectorAll('.char'), {
            opacity: 1,
            y: 0,
            duration: 0.01, 
            stagger: 0.035, 
            ease: 'power3.out',
            delay: delay
        });
    }

    // --- Send Button Animations --- (Removed old timeline logic)
    // const sendButtonHoverTl = gsap.timeline({ paused: true });
    // sendButtonHoverTl.to(sendButton, { scale: 1.1, duration: 0.15, ease: 'power1.out' }); 
    // const sendButtonClickTl = gsap.timeline({ paused: true });
    // sendButtonClickTl.to(sendButton, { scale: 0.9, duration: 0.08, ease: 'power1.inOut' })
    //                  .to(sendButton, { scale: 1, duration: 0.08, ease: 'power1.inOut' }); 
    // sendButton.addEventListener('mouseenter', () => sendButtonHoverTl.play());
    // sendButton.addEventListener('mouseleave', () => sendButtonHoverTl.reverse());

    sendButton.addEventListener('click', () => {
        // The visual change to sending icon will happen in handleSendMessage/proceedWithSending
        handleSendMessage();
    });

    // --- User Input Focus Animation ---
    userInput.addEventListener('focus', () => {
        gsap.to(userInput.parentElement, { 
            borderColor: 'var(--bg-accent)', 
            duration: 0.25, 
            ease: 'power2.out'
        });
    });
    userInput.addEventListener('blur', () => {
        gsap.to(userInput.parentElement, {
            borderColor: 'var(--border-color)',
            duration: 0.25, 
            ease: 'power1.in'
        });
    });

    // --- Chat Logic ---
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            // Visual change to sending icon will happen in handleSendMessage/proceedWithSending
            handleSendMessage();
        }
    });

    let currentModel = 'gemma'; 

    function updateChatHeaderTitle(title) {
        if (chatHeaderTitleElement) {
            chatHeaderTitleElement.textContent = title;
            gsap.from(chatHeaderTitleElement, {opacity:0, y:-10, duration:0.3});
        }
    }

    async function handleChatSelection(chatId, chatTitle) {
        console.log(`Selected chat: ${chatId} - ${chatTitle}`);
        currentActiveChatId = chatId;
        updateChatHeaderTitle(chatTitle);
        chatOutput.innerHTML = ''; // Clear current messages

        if (initialPromptArea && getComputedStyle(initialPromptArea).display !== 'none') {
            gsap.to(initialPromptArea, { opacity: 0, duration: 0.3, onComplete: () => initialPromptArea.style.display = 'none' });
            gsap.to(chatOutput, { opacity: 1, duration: 0.1, delay: 0.2 });
        }

        // Show thinking indicator while fetching messages
        const thinkingLottie = appendLottieThinkingMessage();

        try {
            const response = await fetch(`/api/chats/${chatId}/messages`);
            if (!response.ok) {
                const errData = await response.json().catch(() => ({error: `HTTP error! Status: ${response.status}`}));
                throw new Error(errData.error || `Failed to fetch messages for chat ${chatId}`);
            }
            const data = await response.json();
            if (thinkingLottie) removeLottieThinkingMessage(thinkingLottie);

            if (data.messages && data.messages.length > 0) {
                data.messages.forEach(msg => {
                    appendMessage(msg.content, msg.sender === 'user' ? 'user-message' : 'bot-message');
                });
            } else {
                // appendMessage('No messages in this chat yet.', 'bot-message'); // Optional: if you want to show a placeholder
                 console.log("No messages in this chat or chat is new.");
            }
            if (data.title) updateChatHeaderTitle(data.title); // Update title again from fetch if it changed

        } catch (error) {
            console.error('Error fetching chat messages:', error);
            if (thinkingLottie) removeLottieThinkingMessage(thinkingLottie);
            appendMessage(`Error loading chat: ${error.message}`, 'bot-message');
        }
        userInput.focus();
    }

    function handleSendMessage() {
        const messageText = userInput.value.trim();
        if (messageText === '') return;

        if (!currentActiveChatId) {
            // If no chat is active, implicitly create a new one without prompting for a name.
            console.log("No active chat. Attempting to create a new one silently.");
            // Use the first part of the message as a potential title, or default to "New Chat"
            const firstMessagePart = messageText.substring(0, 30); // Take first 30 chars for a potential title
            const defaultTitle = firstMessagePart + (messageText.length > 30 ? "..." : "") || "New Chat";

            handleCreateNewChat(defaultTitle, null, false) // promptForName = false
                .then(newChatInfo => { // handleCreateNewChat will now return {id, title}
                    if(newChatInfo && newChatInfo.id) {
                        currentActiveChatId = newChatInfo.id; 
                        updateChatHeaderTitle(newChatInfo.title); // Update header with actual title used
                        proceedWithSending(messageText); // Now send the original message
                    } else {
                        appendMessage("Could not start a new chat. Please try again.", "bot-message");
                    }
                });
            return; 
        }
        proceedWithSending(messageText);
    }

    function proceedWithSending(messageText) {
        appendMessage(messageText, 'user-message');
        userInput.value = '';
        userInput.style.height = 'auto'; 
        sendButton.classList.remove('active'); 
        userInput.focus();

        if (initialPromptArea && getComputedStyle(initialPromptArea).display !== 'none') {
            gsap.to(initialPromptArea, { opacity: 0, duration: 0.3, onComplete: () => initialPromptArea.style.display = 'none' });
            gsap.to(chatOutput, { opacity: 1, duration: 0.1, delay: 0.2 }); 
        }

        // Create a placeholder message for the bot's response that will be updated with stream.
        const botMessageWrapper = appendMessage('', 'bot-message', true); // true for isThinking initially
        const botMessageParagraph = botMessageWrapper.querySelector('p');
        const thinkingDots = botMessageWrapper.querySelector('.thinking-dots');
        if(thinkingDots) thinkingDots.style.display = 'inline'; // Show thinking dots
        if(botMessageParagraph) botMessageParagraph.innerHTML = ''; // Clear any default thinking text

        const payload = {
            message: messageText,
            model_type: currentModel,
            chat_id: currentActiveChatId
        };

        // Change send button to sending state (FontAwesome)
        const sendButtonIconContainer = document.getElementById('lottie-send-button-icon');
        if (sendButtonIconContainer) {
            sendButtonIconContainer.innerHTML = '<i class="fas fa-spinner fa-pulse"></i>';
        }
        sendButton.disabled = true; // Disable button while sending

        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        })
        .then(response => {
            if (!response.ok) {
                // Attempt to read error body as JSON, then text, then throw generic error
                return response.json()
                    .then(err => { throw new Error(err.error || `HTTP error! Status: ${response.status}`); })
                    .catch(() => response.text().then(textErr => { throw new Error(textErr || `HTTP error! Status: ${response.status}`); }));
            }
            
            // Handle streaming response
            if (!response.body) {
                throw new Error('ReadableStream not available.');
            }
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let accumulatedText = '';
            let firstChunkReceived = false;

            function push() {
                reader.read().then(({ done, value }) => {
                    if (done) {
                        console.log('Stream complete');
                        if(thinkingDots) thinkingDots.style.display = 'none';

                        // Add copy button for the completed bot message if text was received
                        if (accumulatedText.trim() !== '' && botMessageWrapper) {
                            const messageContentDiv = botMessageWrapper.querySelector('.message-content');
                            if (messageContentDiv && !messageContentDiv.querySelector('.copy-btn')) { // Check if not already added
                                const copyButton = document.createElement('button');
                                copyButton.classList.add('icon-btn', 'copy-btn');
                                copyButton.title = 'Copy message';
                                copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                                copyButton.addEventListener('click', () => {
                                    navigator.clipboard.writeText(accumulatedText) // Use accumulatedText (raw markdown)
                                        .then(() => {
                                            copyButton.innerHTML = '<i class="fas fa-check"></i>';
                                            setTimeout(() => { copyButton.innerHTML = '<i class="fas fa-copy"></i>'; }, 2000);
                                        })
                                        .catch(err => {
                                            console.error('Failed to copy streamed text: ', err);
                                            // Basic fallback for HTTP or if clipboard API fails
                                            try {
                                                const textArea = document.createElement("textarea");
                                                textArea.value = accumulatedText;
                                                document.body.appendChild(textArea);
                                                textArea.focus();
                                                textArea.select();
                                                document.execCommand('copy');
                                                document.body.removeChild(textArea);
                                                copyButton.innerHTML = '<i class="fas fa-check"></i>';
                                                setTimeout(() => { copyButton.innerHTML = '<i class="fas fa-copy"></i>'; }, 2000);
                                            } catch (execErr) {
                                                console.error('Fallback copy failed for streamed text: ', execErr);
                                                alert('Failed to copy. Please copy manually.');
                                            }
                                        });
                                });
                                messageContentDiv.appendChild(copyButton); // Append to message content div
                            }
                        }

                        fetchAndRenderSidebarData(false);
                        if (sendButtonIconContainer) {
                            sendButtonIconContainer.innerHTML = '<i class="fas fa-paper-plane"></i>'; 
                        }
                        sendButton.disabled = false;
                        userInput.focus();
                        return;
                    }

                    if (!firstChunkReceived) {
                        if(thinkingDots) thinkingDots.style.display = 'none'; // Hide thinking dots
                        // Remove the initial thinking state from the bot message wrapper if it had one
                        botMessageWrapper.classList.remove('bot-thinking');
                        firstChunkReceived = true;
                    }

                    const chunkText = decoder.decode(value, { stream: true });
                    accumulatedText += chunkText;
                    
                    // Safely parse and update HTML. `safeMarkdownParse` is assumed to handle partial HTML well enough or just update text.
                    // For a smoother animation, the `animateTextByWord` would need significant changes to support appending.
                    // For now, direct update:
                    if (botMessageParagraph) {
                        // If using marked.js, it needs full content to parse correctly.
                        // So, we update with the full accumulated text each time.
                        botMessageParagraph.innerHTML = safeMarkdownParse(accumulatedText);
                    }
                    // Only auto-scroll if user is already at (or very near) the bottom
                    const isAtBottom = (chatViewContainer.scrollHeight - chatViewContainer.scrollTop - chatViewContainer.clientHeight) < 40;
                    if (isAtBottom) {
                        chatViewContainer.scrollTop = chatViewContainer.scrollHeight;
                    }
                    push(); // Read the next chunk
                }).catch(error => {
                    console.error('Error reading stream:', error);
                    if(thinkingDots) thinkingDots.style.display = 'none';
                    if (botMessageParagraph) botMessageParagraph.innerHTML += `<br><span class="error-text">Stream error: ${error.message}</span>`;
                    // Re-enable send button and set icon on error too
                    if (sendButtonIconContainer) {
                        sendButtonIconContainer.innerHTML = '<i class="fas fa-paper-plane"></i>'; 
                    }
                    sendButton.disabled = false;
                    userInput.focus();
                });
            }
            push(); // Start reading the stream
        })
        .catch(error => {
            console.error('Fetch error in proceedWithSending:', error);
            if(thinkingDots) thinkingDots.style.display = 'none'; // Hide dots on error
            // Update the placeholder message with error
            if (botMessageParagraph) botMessageParagraph.innerHTML = `<span class="error-text">Error: ${error.message}</span>`;
            botMessageWrapper.classList.remove('bot-thinking');
            // Re-enable send button and set icon
            if (sendButtonIconContainer) {
                sendButtonIconContainer.innerHTML = '<i class="fas fa-paper-plane"></i>';
            }
            sendButton.disabled = false;
            userInput.focus(); // Focus back on input
        });
        // Note: The .finally block from the original code is effectively handled by the stream completion or error handling
    }

    function appendLottieThinkingMessage() {
        const messageWrapper = document.createElement('div');
        messageWrapper.classList.add('message', 'bot-message', 'bot-thinking-lottie');
        const avatar = document.createElement('div');
        avatar.classList.add('message-avatar');
        const botAvatarIconContainer = document.createElement('div');
        botAvatarIconContainer.style.width = '32px';
        botAvatarIconContainer.style.height = '32px';
        // Use FontAwesome for bot avatar in thinking message
        botAvatarIconContainer.innerHTML = '<i class="fas fa-robot fa-lg"></i>'; 
        // Center icon if needed with CSS on .message-avatar or direct styles here
        botAvatarIconContainer.style.display = 'flex';
        botAvatarIconContainer.style.alignItems = 'center';
        botAvatarIconContainer.style.justifyContent = 'center';

        avatar.appendChild(botAvatarIconContainer);
        messageWrapper.appendChild(avatar);
        const messageContentDiv = document.createElement('div');
        messageContentDiv.classList.add('message-content');
        const lottieSpinnerContainer = document.createElement('div');
        lottieSpinnerContainer.id = 'lottie-thinking-indicator';
        lottieSpinnerContainer.style.width = '50px';
        lottieSpinnerContainer.style.height = '30px';
        messageContentDiv.appendChild(lottieSpinnerContainer);
        messageWrapper.appendChild(messageContentDiv);
        chatOutput.appendChild(messageWrapper);
        if (activeThinkingAnimation) { 
            activeThinkingAnimation.destroy(); 
        }
        activeThinkingAnimation = loadLottieAnimation(
            lottieSpinnerContainer, 
            'https://lottie.host/5f1a9f74-8a83-4792-b532-637297389783/0kTn5Z22aW.json', 
            true, true, 'svg', { animationID: 'thinking-spinner-animation' } 
        );
        gsap.fromTo(messageWrapper, { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.5, ease: 'power2.out', delay: 0.1 });
        chatViewContainer.scrollTop = chatViewContainer.scrollHeight;
        return messageWrapper; 
    }

    function removeLottieThinkingMessage(thinkingLottieMessageDiv) {
        if (activeThinkingAnimation) { 
            activeThinkingAnimation.destroy();
            activeThinkingAnimation = null;
            loadedLottieAnimations.delete('thinking-spinner-animation'); 
        }
        if (thinkingLottieMessageDiv) {
            const lottieAvatarsInThinkingMsg = thinkingLottieMessageDiv.querySelectorAll('.message-avatar div');
            lottieAvatarsInThinkingMsg.forEach(container => {
                loadedLottieAnimations.forEach((anim, id) => {
                    if (anim.wrapper === container) {
                        anim.destroy();
                        loadedLottieAnimations.delete(id);
                    }
                });
            });
            gsap.to(thinkingLottieMessageDiv, {opacity: 0, y: -10, duration: 0.3, onComplete: () => thinkingLottieMessageDiv.remove()});
        }
    }

    function appendMessage(text, type, isThinking = false) {
        console.log(`[DEBUG] appendMessage called. Type: ${type}, Text Length: ${text.length}, isThinking: ${isThinking}`);
        if (initialPromptArea && getComputedStyle(initialPromptArea).display !== 'none') {
            initialPromptArea.style.display = 'none';
            chatOutput.style.opacity = 1; // Make chat output visible
        }

        const messageWrapper = document.createElement('div');
        messageWrapper.classList.add('message', type);
        messageWrapper.id = `message-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;

        const avatar = document.createElement('div');
        avatar.classList.add('message-avatar');
        const avatarIconContainer = document.createElement('div');
        avatarIconContainer.style.width = '32px'; 
        avatarIconContainer.style.height = '32px';
        avatarIconContainer.style.display = 'flex';
        avatarIconContainer.style.alignItems = 'center';
        avatarIconContainer.style.justifyContent = 'center';

        if (type === 'bot-message') {
            avatarIconContainer.innerHTML = '<i class="fas fa-robot fa-lg"></i>';
        } else { 
            avatarIconContainer.innerHTML = '<i class="fas fa-user-astronaut fa-lg"></i>';
        }
        avatar.appendChild(avatarIconContainer);
        messageWrapper.appendChild(avatar);

        const messageContentDiv = document.createElement('div');
        messageContentDiv.classList.add('message-content');
        
        const messageParagraph = document.createElement('p');
        let animationTimeline = null; 

        if (isThinking && type === 'bot-message') { 
             const dotsContainer = document.createElement('span');
             dotsContainer.classList.add('thinking-dots');
             dotsContainer.innerHTML = '<span>.</span><span>.</span><span>.</span>';
             dotsContainer.style.display = 'none'; 
             messageParagraph.appendChild(dotsContainer);
        } else if (type === 'bot-message' && !isThinking) { 
            const textToParse = (typeof text === 'string' || text instanceof String) ? text : '';
            let htmlOutput = '';
            try {
                if (typeof marked === 'undefined') throw new Error('marked is not defined');
                htmlOutput = marked.parse(textToParse, { sanitize: false, mangle: false });
             } catch (e) {
                htmlOutput = safeMarkdownParse(textToParse);
             }
            animationTimeline = animateTextByWord(messageParagraph, htmlOutput);

            // Add copy button for bot messages
            const copyButton = document.createElement('button');
            copyButton.classList.add('icon-btn', 'copy-btn');
            copyButton.title = 'Copy message';
            copyButton.innerHTML = '<i class="fas fa-copy"></i>';
            copyButton.addEventListener('click', () => {
                navigator.clipboard.writeText(textToParse) // Use textToParse which is the raw markdown
                    .then(() => {
                        copyButton.innerHTML = '<i class="fas fa-check"></i>';
                        setTimeout(() => {
                            copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                        }, 2000);
                    })
                    .catch(err => {
                        console.error('Failed to copy text: ', err);
                        // Basic fallback for HTTP or if clipboard API fails
                        try {
                            const textArea = document.createElement("textarea");
                            textArea.value = textToParse;
                            document.body.appendChild(textArea);
                            textArea.focus();
                            textArea.select();
                            document.execCommand('copy');
                            document.body.removeChild(textArea);
                            copyButton.innerHTML = '<i class="fas fa-check"></i>';
                            setTimeout(() => {
                                copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                            }, 2000);
                        } catch (execErr) {
                            console.error('Fallback copy failed: ', execErr);
                            alert('Failed to copy message. Please copy manually.');
                        }
                    });
            });
            // Prepend copy button to messageContentDiv for better layout control with CSS
            messageContentDiv.appendChild(copyButton); 

        } else if (isThinking && !messageWrapper.classList.contains('bot-thinking-lottie')) { 
            messageParagraph.textContent = text; 
            if (messageWrapper.classList.contains('bot-thinking')) { 
                const dotsContainer = document.createElement('span');
                dotsContainer.classList.add('thinking-dots');
                dotsContainer.innerHTML = '<span>.</span><span>.</span><span>.</span>';
                dotsContainer.style.display = 'none'; 
                messageParagraph.appendChild(dotsContainer);
            }
        } else if (type === 'user-message') { 
            messageParagraph.textContent = text; 
             // Add copy button for user messages
            const copyButton = document.createElement('button');
            copyButton.classList.add('icon-btn', 'copy-btn');
            copyButton.title = 'Copy message';
            copyButton.innerHTML = '<i class="fas fa-copy"></i>';
            copyButton.addEventListener('click', () => {
                navigator.clipboard.writeText(text) // User message text is already plain
                    .then(() => {
                        copyButton.innerHTML = '<i class="fas fa-check"></i>';
                        setTimeout(() => {
                            copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                        }, 2000);
                    })
                    .catch(err => {
                        console.error('Failed to copy text for user message: ', err);
                         try {
                            const textArea = document.createElement("textarea");
                            textArea.value = text;
                            document.body.appendChild(textArea);
                            textArea.focus();
                            textArea.select();
                            document.execCommand('copy');
                            document.body.removeChild(textArea);
                            copyButton.innerHTML = '<i class="fas fa-check"></i>';
                            setTimeout(() => {
                                copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                            }, 2000);
                        } catch (execErr) {
                            console.error('Fallback copy failed for user message: ', execErr);
                            alert('Failed to copy message. Please copy manually.');
                        }
                    });
            });
            messageContentDiv.appendChild(copyButton);
        } else {
            messageParagraph.innerHTML = text; 
        }

        messageContentDiv.appendChild(messageParagraph);
        messageWrapper.appendChild(messageContentDiv);
        chatOutput.appendChild(messageWrapper);

        gsap.set(messageWrapper, { opacity: 0, y: 15 });
        let scrollDelay = 0.1; 
        const wrapperAppearTl = gsap.timeline();

        if (type === 'bot-message' && !isThinking) {
            const textContentForDelay = messageParagraph.textContent || '';
            scrollDelay = Math.min(1.5, Math.max(0.2, textContentForDelay.length * 0.005 + 0.3)); 
            wrapperAppearTl.to(messageWrapper, {
                opacity: 1,
                y: 0,
                duration: 0.3, 
                ease: 'power2.out',
                onComplete: () => {
                    if (!animationTimeline && messageParagraph.innerHTML.trim() !== '') {
                        if (messageParagraph.style.opacity === '0') messageParagraph.style.opacity = '1';
                    }
                }
            });
        } else {
            wrapperAppearTl.to(messageWrapper, {
                opacity: 1,
                y: 0,
                duration: 0.3,
                ease: 'power2.out'
            });
        }

        gsap.delayedCall(scrollDelay, () => {
            if (chatViewContainer) {
                chatViewContainer.scrollTop = chatViewContainer.scrollHeight;
            }
        });

        if (sidebar && !sidebar.classList.contains('collapsed') && mainContentSidebarToggleBtn) {
            const toggleButtonElement = mainContentSidebarToggleBtn.closest('.icon-btn');
            if (toggleButtonElement) {
                sidebar.classList.add('collapsed'); 
                gsap.to(sidebar, { 
                    width: 0, 
                    padding: '0px', 
                    opacity: 0,
                    duration: 0.5,
                    ease: 'expo.inOut'
                });
                if (mainContentSidebarToggleBtn) {
                    mainContentSidebarToggleBtn.classList.remove('fa-chevron-left');
                    mainContentSidebarToggleBtn.classList.add('fa-bars');
                }
            }
        }
        return messageWrapper; 
    }

    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = (userInput.scrollHeight) + 'px';
        if (userInput.value.trim() !== '') {
            sendButton.classList.add('active');
        } else {
            sendButton.classList.remove('active');
        }
    });

    if (userInput.value.trim() !== '') {
        sendButton.classList.add('active');
    } else {
        sendButton.classList.remove('active');
    }

    suggestionCards.forEach(card => {
        card.addEventListener('click', () => {
            const cardTitle = card.querySelector('h3').textContent;
            userInput.value = `Tell me more about ${cardTitle.toLowerCase()}`;
            userInput.focus();
        });
    });

    // --- Sidebar Interactivity & Dynamic Data --- 
    async function fetchAndRenderSidebarData(animate = true) {
        try {
            const response = await fetch('/api/sidebar-data');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();

            window.currentFoldersData = data.folders || [];

            renderFolders(data.folders || [], animate);
            renderDirectChats(data.uncategorized_chats || [], animate);

            attachSectionToggleListeners(); 

        } catch (error) {
            console.error("Failed to fetch sidebar data:", error);
            if (foldersListUL) foldersListUL.innerHTML = '<li>Error loading folders.</li>';
            if (directChatsListUL) directChatsListUL.innerHTML = '<li>Error loading chats.</li>';
        }
    }

    function renderFolders(folders, animate) {
        if (!foldersListUL) return;
        foldersListUL.innerHTML = ''; // Clear existing
        if (folders.length === 0) {
            foldersListUL.innerHTML = '<li class="empty-state-sidebar">No folders yet.</li>';
        }
        folders.forEach(folder => {
            const folderLi = document.createElement('li');
            folderLi.className = 'folder-item'; // Add a class for styling if needed
            folderLi.innerHTML = `
                <div class="nav-section-header folder-header">
                    <a href="#" class="folder-name-link">
                        <div class="lottie-sidebar-icon" data-lottie-path="https://lottie.host/b01e0689-448a-4f7c-912e-7e985398f0e3/N1GQB0jNVu.json"></div>
                        <span>${escapeHTML(folder.name)}</span>
                    </a>
                    <div class="folder-action-buttons">
                        <button class="icon-btn icon-btn-sm add-chat-to-folder-btn" data-folder-id="${folder.id}" title="New chat in this folder"><i class="fas fa-plus"></i></button>
                        <button class="icon-btn icon-btn-sm folder-options-btn" data-folder-id="${folder.id}" title="Folder options"><i class="fas fa-ellipsis-h"></i></button>
                    </div>
                </div>
                <ul class="chats-in-folder-list" data-folder-id="${folder.id}">
                    ${(folder.chats && folder.chats.length > 0) ? folder.chats.map(chat => `
                        <li data-chat-id="${chat.id}" data-chat-title="${escapeHTML(chat.title)}">
                            <a href="#">
                                <div class="lottie-sidebar-icon" data-lottie-path="https://lottie.host/9e9a2c8c-0071-49c6-a4e5-797916f75483/Lp8xDLs5sD.json"></div>
                                <span class="chat-title">${escapeHTML(chat.title)}</span>
                                ${chat.last_snippet ? `<p class="chat-snippet">${escapeHTML(chat.last_snippet)}</p>` : ''}
                            </a>
                            <button class="icon-btn-sm chat-options-btn" data-chat-id="${chat.id}" title="Chat options"><i class="fas fa-ellipsis-h"></i></button>
                        </li>
                    `).join('') : '<li class="empty-state-sidebar"><em>No chats in this folder.</em></li>'}
                </ul>
            `;
            foldersListUL.appendChild(folderLi);
            // Add Lottie for folder icon itself
            const folderLottieIcon = folderLi.querySelector('.folder-name-link .lottie-sidebar-icon');
            console.log(`[DEBUG Folder ${folder.id}] Name: "${folder.name}", Options button found:`, folderLottieIcon);
            if(folderLottieIcon) loadLottieAnimation(folderLottieIcon, folderLottieIcon.dataset.lottiePath, true, false, 'svg');

            // Event listener for the folder options button:
            const folderOptionsBtn = folderLi.querySelector('.folder-options-btn');
            // DEBUG LINE VVV
            console.log(`[DEBUG Folder Options Button] For folder '${folder.name}' (ID: ${folder.id}), button element:`, folderOptionsBtn);
            // DEBUG LINE ^^^ 
            if (folderOptionsBtn) {
                folderOptionsBtn.addEventListener('click', async (e) => {
                    e.stopPropagation(); 
                    const folderId = folderOptionsBtn.dataset.folderId;
                    // folder.name is available from the forEach context
                    const action = await showActionChoiceModal(
                        'Folder Options',
                        `Selected: ${escapeHTML(folder.name)}`,
                        "Rename Folder",
                        "Delete Folder"
                    );

                    if (action === 'action1') { // Rename Folder
                        handleRenameFolder(folderId, folder.name);
                    } else if (action === 'action2') { // Delete Folder
                        handleDeleteFolder(folderId, folder.name);
                    }
                });
            }

            // Add Lottie for chat icons within this folder
            folderLi.querySelectorAll('.chats-in-folder-list .lottie-sidebar-icon').forEach(iconContainer => {
                const path = iconContainer.dataset.lottiePath;
                if (path) {
                    const anim = loadLottieAnimation(iconContainer, path, true, false, 'svg');
                    if (anim) {
                        const listItem = iconContainer.closest('li[data-chat-id]');
                        if (listItem) {
                            listItem.addEventListener('mouseenter', () => anim.play());
                            listItem.addEventListener('mouseleave', () => anim.pause());
                        }
                    }
                }
            });

            // Attach click listeners for chats within this folder
            folderLi.querySelectorAll('.chats-in-folder-list li[data-chat-id]').forEach(chatLi => {
                chatLi.addEventListener('click', (e) => {
                    e.preventDefault();
                    // Prevent chat selection if clicking on options button inside
                    if (e.target.closest('.chat-options-btn')) {
                        return;
                    }
                    const chatId = chatLi.dataset.chatId;
                    const chatTitle = chatLi.dataset.chatTitle;
                    handleChatSelection(chatId, chatTitle);
                });

                // Handle chat options for chats in folders
                const chatOptionsBtn = chatLi.querySelector('.chat-options-btn');
                if(chatOptionsBtn){
                    chatOptionsBtn.addEventListener('click', async (e) => {
                e.stopPropagation(); 
                        const chatId = chatOptionsBtn.dataset.chatId;
                        const chatTitle = chatLi.dataset.chatTitle; // Assuming data-chat-title is on the <li>
                        const currentFolderId = folder.id;
                        
                        const action = await showActionChoiceModal(
                            'Chat Options',
                            `Selected: ${escapeHTML(chatTitle)}`,
                            "Rename", 
                            "Move to Folder",
                            "Delete Chat" // Third action for chats
                        );

                        if (action === 'action1') { // Rename
                            handleRenameChat(chatId, chatTitle);
                        } else if (action === 'action2') { // Move
                            handleShowMoveToFolderModal(chatId, currentFolderId);
                        } else if (action === 'action3') { // Delete Chat
                            handleDeleteChat(chatId, chatTitle);
                        }
                    });
                }
            });
            
            // Add click listener for "New chat in folder" button
            const addChatToFolderBtn = folderLi.querySelector('.add-chat-to-folder-btn');
            if(addChatToFolderBtn){
                addChatToFolderBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    const folderId = addChatToFolderBtn.dataset.folderId;
                    await handleCreateNewChat(null, folderId); // Let createNewChat handle selection and refresh
                });
            }

            if (animate) {
                gsap.from(folderLi, { opacity: 0, x: -20, duration: 0.3, delay: folders.indexOf(folder) * 0.05 });
            }
        });
    }

    function renderDirectChats(chats, animate) {
        if (!directChatsListUL) return;
        directChatsListUL.innerHTML = ''; // Clear existing
        if (chats.length === 0) {
             directChatsListUL.innerHTML = '<li class="empty-state-sidebar">No chats.</li>';
        }
        chats.forEach(chat => {
            const chatLi = document.createElement('li');
            chatLi.dataset.chatId = chat.id;
            chatLi.dataset.chatTitle = escapeHTML(chat.title);
            chatLi.innerHTML = `
                <a href="#">
                    <div class="lottie-sidebar-icon" data-lottie-path="https://lottie.host/9e9a2c8c-0071-49c6-a4e5-797916f75483/Lp8xDLs5sD.json"></div>
                    <span class="chat-title">${escapeHTML(chat.title)}</span>
                    ${chat.last_snippet ? `<p class="chat-snippet">${escapeHTML(chat.last_snippet)}</p>` : ''}
                </a>
                <button class="icon-btn-sm chat-options-btn" data-chat-id="${chat.id}" title="Chat options"><i class="fas fa-ellipsis-h"></i></button>
            `;
            chatLi.addEventListener('click', (e) => {
                e.preventDefault();
                 // Prevent chat selection if clicking on options button inside
                if (e.target.closest('.chat-options-btn')) {
                    return;
                }
                handleChatSelection(chat.id, chat.title);
            });

            // Handle chat options for uncategorized chats
            const chatOptionsBtn = chatLi.querySelector('.chat-options-btn');
            if(chatOptionsBtn){
                chatOptionsBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    const chatId = chatOptionsBtn.dataset.chatId;
                    const chatTitle = chatLi.dataset.chatTitle; // Assuming data-chat-title is on the <li>
                    
                    const action = await showActionChoiceModal(
                        'Chat Options',
                        `Selected: ${escapeHTML(chatTitle)}`,
                        "Rename", 
                        "Move to Folder",
                        "Delete Chat" // Third action for chats
                    );

                    if (action === 'action1') { // Rename
                        handleRenameChat(chatId, chatTitle);
                    } else if (action === 'action2') { // Move
                        handleShowMoveToFolderModal(chatId, null); // null for currentFolderId as it's uncategorized
                    } else if (action === 'action3') { // Delete Chat
                        handleDeleteChat(chatId, chatTitle);
                    }
                });
            }

            directChatsListUL.appendChild(chatLi);

            const lottieIcon = chatLi.querySelector('.lottie-sidebar-icon');
            if(lottieIcon) {
                 const anim = loadLottieAnimation(lottieIcon, lottieIcon.dataset.lottiePath, true, false, 'svg');
                 if (anim) {
                    chatLi.addEventListener('mouseenter', () => anim.play());
                    chatLi.addEventListener('mouseleave', () => anim.pause());
                 }
            }
            if (animate) {
                gsap.from(chatLi, { opacity: 0, x: -20, duration: 0.3, delay: chats.indexOf(chat) * 0.05 });
            }
        });
    }

    function escapeHTML(str) {
        if (typeof str !== 'string') return '';
        return str.replace(/[&<>"]/g, function (tag) {
            const charsToReplace = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;'
            };
            return charsToReplace[tag] || tag;
        });
    }
    
    function attachSectionToggleListeners() {
    const toggleButtons = document.querySelectorAll('.sidebar-nav .nav-section-header .icon-btn[data-section-toggle]');
    toggleButtons.forEach(button => {
        // Remove existing listener to prevent multiple attachments if called again
        // A bit hacky; ideally, manage listeners more robustly if elements persist
        const newButton = button.cloneNode(true);
        button.parentNode.replaceChild(newButton, button);

        const listId = newButton.dataset.sectionToggle;
        const listElement = document.getElementById(listId);
        const chevronIcon = newButton.querySelector('i.fa-chevron-down, i.fa-chevron-up');

        if (listElement && chevronIcon) {
            // Check initial state from class or by visibility
            const isInitiallyCollapsed = listElement.classList.contains('collapsed') || getComputedStyle(listElement).height === '0px';
            if (isInitiallyCollapsed) {
                gsap.set(listElement, { height: 0, opacity: 0 });
                chevronIcon.classList.remove('fa-chevron-up');
                chevronIcon.classList.add('fa-chevron-down');
            } else {
                // Set initial state for animation from 'auto' height
                const currentHeight = listElement.offsetHeight; // Get the actual height when not collapsed
                gsap.set(listElement, { height: currentHeight, opacity: 1 });
                chevronIcon.classList.remove('fa-chevron-down');
                chevronIcon.classList.add('fa-chevron-up');
            }

            newButton.addEventListener('click', (e) => {
                e.stopPropagation();
                // Determine the target height for animation *before* toggling the class
                const isCollapsing = !listElement.classList.contains('collapsed'); // about to collapse
                
                // Toggle the class first
                listElement.classList.toggle('collapsed');

                // Animate the height based on whether it's collapsing or expanding
                gsap.to(listElement, {
                    height: isCollapsing ? 0 : 'auto', // Animate to 0 if collapsing, or 'auto' if expanding
                    opacity: isCollapsing ? 0 : 1,
                    duration: 0.35,
                    ease: 'power2.inOut',
                    onComplete: () => {
                         // Optional: Ensure height is 'auto' after expansion for responsiveness
                         if (!isCollapsing) {
                             gsap.set(listElement, { height: 'auto' });
                         }
                    }
                });

                // Update the icon class
                if (isCollapsing) {
                    chevronIcon.classList.remove('fa-chevron-up');
                    chevronIcon.classList.add('fa-chevron-down');
                } else {
                    chevronIcon.classList.remove('fa-chevron-down');
                    chevronIcon.classList.add('fa-chevron-up');
                }
            });
        }
    });
}
        
    

    // --- Custom Input Modal Logic ---
    function showInputModal(title, placeholder, currentVal = '') {
        return new Promise((resolve) => {
            if (!inputModal || !modalTitleEl || !modalInputEl || !modalSubmitBtn || !modalCancelBtn) {
                console.error("Modal elements not found!");
                resolve(null); // Or reject, depending on desired error handling
                return;
            }

            modalTitleEl.textContent = title;
            modalInputEl.placeholder = placeholder;
            modalInputEl.value = currentVal;
            inputModal.style.display = 'flex'; // Show the modal overlay
            setTimeout(() => inputModal.classList.add('visible'), 10); // Trigger transition

            modalInputEl.focus(); // Focus the input field

            const handleSubmit = () => {
                cleanup();
                resolve(modalInputEl.value.trim());
            };

            const handleCancel = () => {
                cleanup();
                resolve(null); // Indicates cancellation
            };
            
            const handleKeydown = (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    handleSubmit();
                } else if (e.key === 'Escape') {
                    handleCancel();
                }
            };

            modalSubmitBtn.onclick = handleSubmit;
            modalCancelBtn.onclick = handleCancel;
            modalInputEl.addEventListener('keydown', handleKeydown);
            // Also close if clicking on overlay, but not on modal content itself
            inputModal.addEventListener('click', (e) => {
                if (e.target === inputModal) { // Clicked on overlay
                    handleCancel();
                }
            });

            function cleanup() {
                inputModal.classList.remove('visible');
                setTimeout(() => inputModal.style.display = 'none', 300); // Hide after transition
                modalSubmitBtn.onclick = null;
                modalCancelBtn.onclick = null;
                modalInputEl.removeEventListener('keydown', handleKeydown);
                inputModal.removeEventListener('click', (e) => {if (e.target === inputModal) handleCancel(); });
            }
        });
    }

    // --- Action Choice Modal --- 
    function showActionChoiceModal(title, message, action1Text, action2Text, action3Text = null) {
        return new Promise((resolve) => {
            if (!actionChoiceModal || !actionChoiceOption1Btn || !actionChoiceOption2Btn || !actionChoiceOption3Btn || !actionChoiceCancelBtn) {
                console.error("Action choice modal elements not found!");
                resolve(null);
                return;
            }
            actionChoiceModalTitle.textContent = title;
            actionChoiceModalMessage.innerHTML = message; // Use innerHTML for escaped chat titles
            
            actionChoiceOption1Btn.textContent = action1Text;
            actionChoiceOption1Btn.style.display = 'block';
            actionChoiceOption2Btn.textContent = action2Text;
            actionChoiceOption2Btn.style.display = 'block';

            if (action3Text) {
                actionChoiceOption3Btn.textContent = action3Text;
                actionChoiceOption3Btn.style.display = 'block';
            } else {
                actionChoiceOption3Btn.style.display = 'none';
            }

            actionChoiceModal.style.display = 'flex';
            setTimeout(() => actionChoiceModal.classList.add('visible'), 10);

            const cleanupAndResolve = (action) => {
                actionChoiceModal.classList.remove('visible');
                setTimeout(() => actionChoiceModal.style.display = 'none', 300);
                actionChoiceOption1Btn.onclick = null;
                actionChoiceOption2Btn.onclick = null;
                actionChoiceOption3Btn.onclick = null;
                actionChoiceCancelBtn.onclick = null;
                actionChoiceModal.removeEventListener('click', overlayClickHandler);
                resolve(action);
            };

            actionChoiceOption1Btn.onclick = () => cleanupAndResolve('action1');
            actionChoiceOption2Btn.onclick = () => cleanupAndResolve('action2');
            actionChoiceOption3Btn.onclick = action3Text ? () => cleanupAndResolve('action3') : null;
            actionChoiceCancelBtn.onclick = () => cleanupAndResolve(null);
            
            const overlayClickHandler = (e) => {
                if (e.target === actionChoiceModal) {
                    cleanupAndResolve(null);
                }
            };
            actionChoiceModal.addEventListener('click', overlayClickHandler);
        });
    }

    // --- Rename Folder Logic ---
    async function handleRenameFolder(folderId, currentName) {
        const newName = await showInputModal("Rename Folder", "Enter new folder name", currentName);
        if (newName && newName.trim() !== '' && newName.trim() !== currentName) {
            try {
                const response = await fetch(`/api/folders/${folderId}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ name: newName.trim() }),
                });
                if (!response.ok) {
                    const errData = await response.json().catch(() => ({error: `HTTP Error ${response.status}`}));
                    throw new Error(errData.error || 'Failed to rename folder');
                }
                fetchAndRenderSidebarData(); // Refresh sidebar
            } catch (error) {
                console.error("Error renaming folder:", error);
                alert(`Could not rename folder: ${error.message}`);
            }
        }
    }

    // --- Rename Chat Logic ---
    async function handleRenameChat(chatId, currentTitle) {
        const newTitle = await showInputModal("Rename Chat", "Enter new chat title", currentTitle);
        if (newTitle && newTitle.trim() !== '' && newTitle.trim() !== currentTitle) {
            try {
                const response = await fetch(`/api/chats/${chatId}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ title: newTitle.trim() }),
                });
                if (!response.ok) {
                    const errData = await response.json().catch(() => ({error: `HTTP Error ${response.status}`}));
                    throw new Error(errData.error || 'Failed to rename chat');
                }
                // If the renamed chat is the currently active one, update the main chat header title
                if (currentActiveChatId === chatId) {
                    updateChatHeaderTitle(newTitle.trim());
                }
                fetchAndRenderSidebarData(); // Refresh sidebar
            } catch (error) {
                console.error("Error renaming chat:", error);
                alert(`Could not rename chat: ${error.message}`);
            }
        }
    }

    // --- Move Chat to Folder Modal Logic ---
    async function handleShowMoveToFolderModal(chatId, currentFolderId) {
        if (!folderSelectModal || !folderSelectList || !folderSelectCancelBtn) {
            console.error("Folder select modal elements not found!");
            return;
        }

        // Fetch fresh folder list to populate the modal
        // Alternatively, use window.currentFoldersData if recently fetched
        let folders = [];
        try {
            const response = await fetch('/api/sidebar-data'); // Or a dedicated /api/folders
            if (!response.ok) throw new Error('Failed to fetch folders for modal');
            const data = await response.json();
            folders = data.folders || [];
        } catch (error) {
            console.error("Error fetching folders for move modal:", error);
            alert("Could not load folders to move the chat. Please try again.");
            return;
        }

        folderSelectList.innerHTML = ''; // Clear previous options

        // Add "Uncategorized" option
        const uncategorizedOption = document.createElement('div');
        uncategorizedOption.className = 'folder-select-option uncategorized-option';
        uncategorizedOption.textContent = 'Uncategorized';
        uncategorizedOption.dataset.folderId = 'null'; // Use string "null" for dataset, will be parsed later
        if (currentFolderId === null) {
            uncategorizedOption.style.fontWeight = 'bold'; // Indicate current location
            uncategorizedOption.style.pointerEvents = 'none'; // Disable if already here
            uncategorizedOption.textContent += " (current)";
        }
        uncategorizedOption.addEventListener('click', () => {
            if (currentFolderId !== null) { // Only move if not already uncategorized
                 moveChatToFolder(chatId, null);
            }
        });
        folderSelectList.appendChild(uncategorizedOption);

        folders.forEach(folder => {
            const option = document.createElement('div');
            option.className = 'folder-select-option';
            option.textContent = folder.name;
            option.dataset.folderId = folder.id;
            if (folder.id === currentFolderId) {
                option.style.fontWeight = 'bold'; // Indicate current location
                option.style.pointerEvents = 'none'; // Disable if already here
                option.textContent += " (current)";
            }
            option.addEventListener('click', () => {
                 if (folder.id !== currentFolderId) { // Only move if not already in this folder
                    moveChatToFolder(chatId, folder.id);
                 }
            });
            folderSelectList.appendChild(option);
        });
        
        if (folders.length === 0 && currentFolderId !== null) {
            // If no other folders exist, and chat is in a folder, only "Uncategorized" will be a real option.
        } else if (folders.length === 0 && currentFolderId === null) {
            folderSelectList.innerHTML = '<p style="padding: 10px; text-align: center; color: var(--text-muted);">No folders available to move to.</p>';
        }


        folderSelectModal.style.display = 'flex';
        setTimeout(() => folderSelectModal.classList.add('visible'), 10);

        const handleCancel = () => {
            folderSelectModal.classList.remove('visible');
            setTimeout(() => folderSelectModal.style.display = 'none', 300);
            folderSelectCancelBtn.onclick = null;
            folderSelectModal.removeEventListener('click', overlayClickHandler);
        };
        
        const overlayClickHandler = (e) => {
            if (e.target === folderSelectModal) {
                handleCancel();
            }
        };

        folderSelectCancelBtn.onclick = handleCancel;
        folderSelectModal.addEventListener('click', overlayClickHandler);
    }

    async function moveChatToFolder(chatId, newFolderId) {
        // newFolderId can be null (string "null" from dataset needs parsing) or an integer
        const targetFolderId = (newFolderId === 'null' || newFolderId === null) ? null : parseInt(newFolderId, 10);

        try {
            const response = await fetch(`/api/chats/${chatId}/folder`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ folder_id: targetFolderId }),
            });
            if (!response.ok) {
                const errData = await response.json().catch(() => ({error: `HTTP Error ${response.status}`}));
                throw new Error(errData.error || 'Failed to move chat');
            }
            // const result = await response.json();
            // console.log(result.message);
            
            // Close the modal
            if (folderSelectModal.classList.contains('visible')) {
                folderSelectModal.classList.remove('visible');
                setTimeout(() => folderSelectModal.style.display = 'none', 300);
            }

            fetchAndRenderSidebarData(); // Refresh sidebar
        } catch (error) {
            console.error("Error moving chat:", error);
            alert(`Could not move chat: ${error.message}`);
        }
    }

    async function handleCreateNewFolder() {
        const folderName = await showInputModal("Create New Folder", "Enter folder name");

        if (folderName && folderName.trim() !== '') {
            try {
                const response = await fetch('/api/folders', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ name: folderName.trim() }),
                });
                if (!response.ok) {
                    const errData = await response.json().catch(()=>({error: `HTTP Error ${response.status}`}));
                    throw new Error(errData.error || 'Failed to create folder');
                }
                // const newFolder = await response.json(); // newFolder.id, newFolder.name
                fetchAndRenderSidebarData(); // Refresh sidebar
            } catch (error) {
                console.error("Error creating folder:", error);
                alert(`Could not create folder: ${error.message}`);
            }
        }
    }

    async function handleCreateNewChat(title = null, folderId = null, promptForName = true) {
        let chatTitle = title;

        if (promptForName && !title) { // Only prompt if promptForName is true AND no title was pre-supplied
            chatTitle = await showInputModal("Create New Chat", "Enter chat name (optional)", "New Chat");
            if (chatTitle === null) return null; // User cancelled
            if (chatTitle.trim() === '') chatTitle = "New Chat"; // Default if empty
        } else if (!title) { // No title and no prompt, use a default
            chatTitle = "New Chat";
        }
        // If title was provided, use it (trimmed)
        chatTitle = chatTitle.trim();

        try {
            const payload = { title: chatTitle }; // Use the determined title
            if (folderId) payload.folder_id = folderId;

            const response = await fetch('/api/chats', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });
            if (!response.ok) {
                 const errData = await response.json().catch(()=>({error: `HTTP Error ${response.status}`}));
                throw new Error(errData.error || 'Failed to create chat');
            }
            const newChat = await response.json();
            currentActiveChatId = newChat.id; 
            updateChatHeaderTitle(newChat.title);
            chatOutput.innerHTML = ''; 
            if (initialPromptArea) {
            initialPromptArea.style.display = 'flex';
            gsap.fromTo(initialPromptArea, {opacity: 0, y: 30}, {opacity: 1, y: 0, duration: 0.6, ease: 'expo.out'});
            }

            fetchAndRenderSidebarData(); 
            userInput.focus();
            return { id: newChat.id, title: newChat.title }; // Return id and actual title
        } catch (error) {
            console.error("Error creating chat:", error);
            alert(`Could not create chat: ${error.message}`);
            return null;
        }
    }

    if (addFolderButton) {
        addFolderButton.addEventListener('click', handleCreateNewFolder);
    }

    if (newChatButton) {
        newChatButton.addEventListener('click', async () => {
            const newChatInfo = await handleCreateNewChat(null, null, true); // Explicitly prompt for name here
            // if (newChatInfo && newChatInfo.id) { // No need to select, handleCreateNewChat makes it active
            // }
            //  else {
            //     // Reset to initial prompt if chat creation failed or was cancelled but UI was already cleared
            //     if (initialPromptArea && getComputedStyle(initialPromptArea).display === 'none') {
            //          initialPromptArea.style.display = 'flex';
            //          gsap.fromTo(initialPromptArea, {opacity: 0, y: 30}, {opacity: 1, y: 0, duration: 0.6, ease: 'expo.out'});
            //     }
            // }
        });
    }

    // Sidebar Toggle functionality 
    if (mainContentSidebarToggleBtn && sidebar) {
        const toggleButtonElement = mainContentSidebarToggleBtn.closest('.icon-btn'); 
        if (toggleButtonElement) {
            toggleButtonElement.addEventListener('click', () => {
                clearTimeout(autoCloseTimer); // Clear any pending auto-close
                if (sidebar.classList.contains('collapsed')) {
                    openSidebar();
                } else {
                    closeSidebar();
                }
            });
        }
    }

    // Model Selection Button Logic
    const gemmaBtn = document.getElementById('gemma-btn');
    const geminiBtn = document.getElementById('gemini-btn');
    const wikiBtn = document.getElementById('wiki-btn'); // Added Wiki button
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const appLayout = document.querySelector('.app-layout'); // Target for theme class

    if (gemmaBtn && geminiBtn && wikiBtn) { // Added wikiBtn to condition
        gemmaBtn.addEventListener('click', () => {
            if (currentModel !== 'gemma') {
                currentModel = 'gemma';
                gemmaBtn.classList.add('active');
                geminiBtn.classList.remove('active');
                wikiBtn.classList.remove('active'); // Deactivate wiki
            }
        });
        geminiBtn.addEventListener('click', () => {
            if (currentModel !== 'gemini') {
                currentModel = 'gemini';
                geminiBtn.classList.add('active');
                gemmaBtn.classList.remove('active');
                wikiBtn.classList.remove('active'); // Deactivate wiki
            }
        });
        wikiBtn.addEventListener('click', () => {
            if (currentModel !== 'wikipedia') {
                currentModel = 'wikipedia'; // Set model to 'wikipedia'
                wikiBtn.classList.add('active');
                gemmaBtn.classList.remove('active');
                geminiBtn.classList.remove('active');
            }
        });
    }

    // Load Initial Lottie Animations 
    const mainPromptLogoContainer = document.getElementById('lottie-main-prompt-logo');
    if (mainPromptLogoContainer) {
        // Use FontAwesome for main prompt logo
        mainPromptLogoContainer.innerHTML = '<i class="fas fa-robot fa-3x"></i>'; 
        // Adjust styles if needed, e.g., for color or if fa-3x isn't enough
        mainPromptLogoContainer.style.display = 'flex';
        mainPromptLogoContainer.style.alignItems = 'center';
        mainPromptLogoContainer.style.justifyContent = 'center';
        // mainPromptLogoContainer.style.color = 'var(--text-muted)'; // Example color
    }
    const newChatIconContainer = document.getElementById('lottie-new-chat-icon');
    if (newChatIconContainer) {
        loadLottieAnimation(
            newChatIconContainer, 
            'https://lottie.host/951a8d19-2dd9-4a71-972f-5f26817770c5/5r2MdBFrqZ.json', 
            true, true, 'svg', { animationID: 'new-chat-icon' }
        );
    }
    const sendButtonIconContainer = document.getElementById('lottie-send-button-icon'); // Keep container, change content

    // Initial load of send button icon (FontAwesome)
    if (sendButtonIconContainer) {
        sendButtonIconContainer.innerHTML = '<i class="fas fa-paper-plane"></i>';
    }
    
    // Initial fetch of sidebar data and setup
    fetchAndRenderSidebarData();
    // Set up initial state for section toggles (might need adjustment after dynamic load)
    // attachSectionToggleListeners(); // Called within fetchAndRenderSidebarData now

    // Load Sidebar Lottie Icons (static ones, dynamic ones handled in render functions)
    const sidebarLottieIcons = document.querySelectorAll('.lottie-sidebar-icon[data-lottie-path]:not(.folder-name-link .lottie-sidebar-icon):not(.chats-in-folder-list .lottie-sidebar-icon)');
    sidebarLottieIcons.forEach(iconContainer => {
        const path = iconContainer.dataset.lottiePath;
        if (path) {
            const anim = loadLottieAnimation(iconContainer, path, true, false, 'svg', {
                animationID: `sidebar-icon-static-${iconContainer.parentElement.textContent.trim() || Math.random()}`
            }); 
            if (anim) {
                const listItemOrLink = iconContainer.closest('li, a');
                if (listItemOrLink) {
                    listItemOrLink.addEventListener('mouseenter', () => anim.play());
                    listItemOrLink.addEventListener('mouseleave', () => anim.pause()); 
                }
            }
        }
    });

    // --- Delete Folder Logic ---
    async function handleDeleteFolder(folderId, folderName) {
        const confirmation = await showInputModal(
            `Delete Folder: ${escapeHTML(folderName)}?`,
            `Type the folder name "${escapeHTML(folderName)}" to confirm deletion. Chats in this folder will become uncategorized.`,
            '' // currentVal is empty for confirmation
        );

        if (confirmation === folderName) {
            try {
                const response = await fetch(`/api/folders/${folderId}`, {
                    method: 'DELETE',
                });
                if (!response.ok) {
                    const errData = await response.json().catch(() => ({error: `HTTP Error ${response.status}`}));
                    throw new Error(errData.error || 'Failed to delete folder');
                }
                fetchAndRenderSidebarData();
            } catch (error) {
                console.error("Error deleting folder:", error);
                alert(`Could not delete folder: ${error.message}`);
            }
        } else if (confirmation !== null) { // User typed something but it wasn't a match
            alert("Folder name did not match. Deletion cancelled.");
        }
        // If confirmation is null, user cancelled the input modal, do nothing.
    }

    // --- Delete Chat Logic ---
    async function handleDeleteChat(chatId, chatTitle) {
         const confirmed = confirm(`Are you sure you want to delete the chat "${escapeHTML(chatTitle)}"? This action cannot be undone.`);
        // For a more integrated UI, replace confirm() with a custom confirmation modal similar to showInputModal or showActionChoiceModal
        // For now, using native confirm for brevity.
        // Example using showActionChoiceModal for confirmation:
        // const action = await showActionChoiceModal(
        //     `Delete Chat: ${escapeHTML(chatTitle)}?`,
        //     "This action cannot be undone.",
        //     "Delete",
        //     "Cancel"
        // );
        // if (action !== 'action1') return; // If not delete, then cancel

        if (confirmed) {
            try {
                const response = await fetch(`/api/chats/${chatId}`, {
                    method: 'DELETE',
                });
                if (!response.ok) {
                    const errData = await response.json().catch(() => ({error: `HTTP Error ${response.status}`}));
                    throw new Error(errData.error || 'Failed to delete chat');
                }
                
                // If the deleted chat was the active one, clear the chat view
                if (currentActiveChatId === chatId) {
                    chatOutput.innerHTML = '';
                    updateChatHeaderTitle('Select a Chat');
                    currentActiveChatId = null;
                    if (initialPromptArea) {
                        initialPromptArea.style.display = 'flex';
                        gsap.fromTo(initialPromptArea, {opacity: 0, y: 30}, {opacity: 1, y: 0, duration: 0.6, ease: 'expo.out'});
                    }
                }
                fetchAndRenderSidebarData();
            } catch (error) {
                console.error("Error deleting chat:", error);
                alert(`Could not delete chat: ${error.message}`);
            }
        }
    }

    const sidebarHoverTrigger = document.querySelector('.sidebar-hover-trigger');
    let autoCloseTimer = null;
    const SIDEBAR_AUTO_CLOSE_DELAY = 100; // milliseconds
    let isMouseInsideSidebar = false;

    // --- Sidebar Open/Close Functions ---
    function openSidebar(isTriggeredByHover = false) {
        if (!sidebar.classList.contains('collapsed')) return; // Already open

        sidebar.classList.remove('collapsed');
        if (sidebarHoverTrigger) sidebarHoverTrigger.style.display = 'none';

        gsap.to(sidebar, { 
            width: 'var(--sidebar-width)', 
            padding: '16px', // Ensure original padding is restored
            opacity: 1,
            duration: 0.4, 
            ease: 'power2.out' 
        });
        if (mainContentSidebarToggleBtn) {
            mainContentSidebarToggleBtn.classList.remove('fa-bars');
            mainContentSidebarToggleBtn.classList.add('fa-chevron-left');
        }
        // If opened by hover, set flag to allow auto-close on mouseleave
        // if (isTriggeredByHover) { // This might not be needed if logic is clean
        // }
    }

    function closeSidebar(isTriggeredByHover = false) {
        if (sidebar.classList.contains('collapsed')) return; // Already collapsed
        
        // Do not close if mouse is flagged as inside (e.g. quickly moved out and back in)
        // This check is a bit redundant if the mouseenter on sidebar clears the timer properly.
        // if (isTriggeredByHover && isMouseInsideSidebar) return; 

        sidebar.classList.add('collapsed');
        if (sidebarHoverTrigger) sidebarHoverTrigger.style.display = 'block';

        gsap.to(sidebar, { 
            width: 0, 
            padding: '0px', 
            opacity: 0,
            duration: 0.4, 
            ease: 'power2.inOut' 
        });
        if (mainContentSidebarToggleBtn) {
            mainContentSidebarToggleBtn.classList.remove('fa-chevron-left');
            mainContentSidebarToggleBtn.classList.add('fa-bars');
        }
    }

    // --- New Hover Logic ---
    if (sidebar) {
        sidebar.addEventListener('mouseenter', () => {
            isMouseInsideSidebar = true;
            clearTimeout(autoCloseTimer); // Cancel auto-close if mouse re-enters
        });

        sidebar.addEventListener('mouseleave', () => {
            isMouseInsideSidebar = false;
            // Only start auto-close timer if sidebar is NOT collapsed (i.e., it's open)
            if (!sidebar.classList.contains('collapsed')) {
                autoCloseTimer = setTimeout(() => {
                    // Check again if mouse isn't back inside before closing
                    if (!isMouseInsideSidebar) {
                        closeSidebar(true);
                    }
                }, SIDEBAR_AUTO_CLOSE_DELAY);
            }
        });
    }

    if (sidebarHoverTrigger) {
        sidebarHoverTrigger.addEventListener('mouseenter', () => {
            if (sidebar.classList.contains('collapsed')) {
                openSidebar(true);
            }
        });
        // No mouseleave needed for the trigger itself, as sidebar's mouseleave handles closure.
    }

    // Initial state for hover trigger based on sidebar state
    if (sidebar && sidebarHoverTrigger) {
        if (sidebar.classList.contains('collapsed')) {
            sidebarHoverTrigger.style.display = 'block';
        } else {
            sidebarHoverTrigger.style.display = 'none';
        }
    }

    // --- Theme Toggle Logic ---
    function applyTheme(theme) {
        if (!appLayout || !themeToggleBtn) return;
        const icon = themeToggleBtn.querySelector('i');
        if (theme === 'light') {
            appLayout.classList.add('light-mode');
            if (icon) {
                icon.classList.remove('fa-moon');
                icon.classList.add('fa-sun');
            }
            localStorage.setItem('theme', 'light');
        } else {
            appLayout.classList.remove('light-mode');
            if (icon) {
                icon.classList.remove('fa-sun');
                icon.classList.add('fa-moon');
            }
            localStorage.setItem('theme', 'dark');
        }
    }

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            if (appLayout.classList.contains('light-mode')) {
                applyTheme('dark');
            } else {
                applyTheme('light');
            }
        });
    }

    // Load saved theme on page load
    const savedTheme = localStorage.getItem('theme') || 'dark'; // Default to dark
    applyTheme(savedTheme);

    // --- Export Chat Functionality ---
    if (exportChatBtn) {
        // Click animation for export button
        const exportClickTl = gsap.timeline({ paused: true });
        exportClickTl.to(exportChatBtn, { scale: 0.9, duration: 0.1, ease: 'power1.inOut' })
                     .to(exportChatBtn, { scale: 1, duration: 0.1, ease: 'power1.inOut' });

        exportChatBtn.addEventListener('click', () => {
            exportClickTl.restart(); // Play click animation
            exportCurrentChat();
        });
    }

    async function exportCurrentChat() {
        if (!currentActiveChatId) {
            alert("Please select a chat to export.");
            return;
        }

        try {
            // Fetch all messages for the current chat
            const response = await fetch(`/api/chats/${currentActiveChatId}/messages`);
            if (!response.ok) {
                const errData = await response.json().catch(() => ({ error: `HTTP error! Status: ${response.status}` }));
                throw new Error(errData.error || `Failed to fetch messages for chat ${currentActiveChatId} for export.`);
            }
            const data = await response.json();

            if (!data.messages || data.messages.length === 0) {
                alert("No messages in the current chat to export.");
                return;
            }

            let chatContent = `Chat Title: ${data.title || chatHeaderTitleElement.textContent}\n\n`; // Use fetched title if available
            data.messages.forEach(msg => {
                const sender = msg.sender === 'user' ? "User" : "Bot";
                // Assuming msg.content contains the raw (markdown or plain) text
                const text = msg.content || '[empty message]';
                chatContent += `${sender}: ${text}\n------------------------------------\n`;
            });

            const blob = new Blob([chatContent], { type: 'text/plain;charset=utf-8' });
            const chatTitleForFile = (data.title || chatHeaderTitleElement.textContent).replace(/[^a-z0-9]/gi, '_').toLowerCase() || 'chat_export';
            const filename = `${chatTitleForFile}.txt`;
            
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(link.href);

        } catch (error) {
            console.error('Error exporting chat:', error);
            alert(`Could not export chat: ${error.message}`);
        }
    }

    // --- Clear All Chats Functionality ---
    if (clearAllChatsBtn) {
        clearAllChatsBtn.addEventListener('click', async () => {
            const confirmed = confirm("Are you sure you want to delete ALL chats? This action cannot be undone.");
            if (confirmed) {
                try {
                    const response = await fetch('/api/chats/clear-all', {
                        method: 'DELETE',
                    });
                    if (!response.ok) {
                        const errData = await response.json().catch(() => ({ error: `HTTP Error ${response.status}` }));
                        throw new Error(errData.error || 'Failed to clear all chats');
                    }
                    // Successfully cleared chats
                    chatOutput.innerHTML = ''; // Clear current chat view
                    updateChatHeaderTitle('Select a Chat'); // Reset header
                    currentActiveChatId = null; // Reset active chat ID
                    if (initialPromptArea) { // Show initial prompt area
                        initialPromptArea.style.display = 'flex';
                        gsap.fromTo(initialPromptArea, {opacity: 0, y: 30}, {opacity: 1, y: 0, duration: 0.6, ease: 'expo.out'});
                    }
                    fetchAndRenderSidebarData(); // Refresh sidebar (should be empty or show default state)
                    alert("All chats have been cleared.");
                } catch (error) {
                    console.error("Error clearing all chats:", error);
                    alert(`Could not clear all chats: ${error.message}`);
                }
            }
        });
    }

});