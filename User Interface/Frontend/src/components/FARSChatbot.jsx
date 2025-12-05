import React, { useState, useEffect, useRef } from "react"

export default function FARSChatbot() {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      type: "bot",
      text: "Hi! Ask me anything about FARS accident data."
    }
  ])
  const [inputValue, setInputValue] = useState(
    ""
  )
  const [isDisabled, setIsDisabled] = useState(false)
  const [isThinking, setIsThinking] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isThinking])

  const handleSend = async () => {
    if (!inputValue.trim() || isDisabled) return;

    const userMessage = {
      id: Date.now(),
      type: "user",
      text: inputValue.trim()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue("");
    setIsDisabled(true);
    setIsThinking(true);

    try {
      const response = await fetch("http://127.0.0.1:5000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMessage.text })
      });

      // NOTE: This block handles HTTP errors (4xx, 5xx), which often include SQL errors 
      // where the server responded but with an error status.
      if (!response.ok) {
        // Attempt to read the error message from the response body if available,
        // otherwise default to a generic error message.
        let errorDetail = "Server responded with an error status.";
        try {
          const errorJson = await response.json();
          // Assuming the backend sends an 'error' field on failure
          if (errorJson.error) {
            errorDetail = errorJson.error;
          } else if (response.statusText) {
            errorDetail = `HTTP ${response.status}: ${response.statusText}`;
          }
        } catch (e) {
          // If response body isn't JSON, use status text
          if (response.statusText) {
            errorDetail = `HTTP ${response.status}: ${response.statusText}`;
          }
        }
        
        // This is the message for a query/SQL execution failure
        setMessages(prev => [
          ...prev,
          {
            id: Date.now() + 1,
            type: "bot",
            text: `⚠️ **Query Failed:** There was an issue processing your request. ${errorDetail}`
          }
        ]);
        
        // Skip the rest of the 'try' block on error
        setIsThinking(false);
        setIsDisabled(false);
        return; 
      }
      // END of !response.ok handling
      

      const data = await response.json();

      let botText = "";

      // If backend returns a natural-language answer, show it first.
      if (data.answer && typeof data.answer === "string") {
        botText += data.answer.trim() + "\n\n";
      }

      const botMessage = {
        id: Date.now() + 1,
        type: "bot",
        text: botText.trim()
      };

      setMessages(prev => [...prev, botMessage]);

    } catch (err) {
      // NOTE: This catch block now primarily handles network errors (e.g., server offline, CORS issues) 
      // or issues parsing the final JSON response, but it no longer catches the 'Backend error' 
      // thrown by the '!response.ok' check above.
      setMessages(prev => [
        ...prev,
        {
          id: Date.now() + 1,
          type: "bot",
          text: `❌ **Connection Error:** Could not connect to the backend server (http://127.0.0.1:5000). Please check the server status. Detail: ${err.message}`
        }
      ]);
    }

    setIsThinking(false);
    setIsDisabled(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !isDisabled) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 p-4">
      <style>
        {`
          @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60%, 100% { content: '...'; }
          }
          
          .thinking-dots::after {
            content: '.';
            animation: dots 1.5s infinite;
          }
        `}
      </style>

      <div className="w-full max-w-2xl bg-white rounded-2xl shadow-xl overflow-hidden">
        {/* Header */}
        <div className="bg-maroon px-6 py-5" style={{ backgroundColor: '#630031' }}>
          <h1 className="text-white text-2xl font-bold">
            FARS Conversational Query
          </h1>
        </div>

        {/* Messages area */}
        <div className="h-96 overflow-y-auto p-6 space-y-4 bg-gray-50">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-xs md:max-w-md px-5 py-3 rounded-2xl shadow-sm ${
                  message.type === "user" ? "bg-orange-600 text-white" : "text-white"
                }`}
                style={message.type === "bot" ? { backgroundColor: '#630031' } : { backgroundColor: '#CF5A00' }}
              >
                <div className="text-sm leading-relaxed">
                  {message.text.split("\n").map((line, i) => (
                    <div key={i}>{line}</div>
                  ))}
                </div>
              </div>
            </div>
          ))}

          {isThinking && (
            <div className="flex justify-start">
              <div className="max-w-xs md:max-w-md px-5 py-3 rounded-2xl shadow-sm text-white" style={{ backgroundColor: '#630031' }}>
                <p className="text-sm thinking-dots">Analyzing data</p>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>


        {/* Input area */}
        <div className="border-t border-gray-200 p-4 bg-white">
          <div className="flex gap-3 items-end">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isDisabled}
              className="flex-1 px-5 py-3 border-2 rounded-full text-gray-800 focus:outline-none focus:ring-2 disabled:bg-gray-100 disabled:cursor-not-allowed text-sm"
              style={{ borderColor: '#630031', focusRingColor: '#CF5A00' }}
              placeholder="Type your query..."
            />
            <button
              onClick={handleSend}
              disabled={isDisabled}
              className="px-8 py-3 text-white rounded-full font-semibold hover:opacity-90 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all text-sm whitespace-nowrap"
              style={{ backgroundColor: isDisabled ? '#cccccc' : '#CF5A00' }}
            >
              {isDisabled ? "Sending..." : "Send"}
            </button>
          </div>
          <p className="text-center text-xs text-gray-500 mt-3">
            Supported by the Fatality Analysis Reporting System (FARS)
          </p>
        </div>
      </div>
    </div>
  )
}
