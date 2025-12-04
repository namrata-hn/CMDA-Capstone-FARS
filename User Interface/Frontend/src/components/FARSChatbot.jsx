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

      if (!response.ok) throw new Error("Backend error");

      const data = await response.json();

      let botText = "";

      // 1️⃣ If backend returns a natural-language answer, show it first.
      if (data.answer && typeof data.answer === "string") {
        botText += data.answer.trim() + "\n\n";
      }

      // 2️⃣ Then show the table (columns/rows)
      /* if (data.columns && data.rows) {
        if (data.rows.length === 1) {
          // single-row → inline result
          botText += data.rows[0].join(" | ");
        } else {
          const header = data.columns.join(" | ");
          const rows = data.rows.map(row => row.join(" | ")).join("\n");
          botText += header + "\n" + rows;
        }
      } else {
        botText += "No data returned.";
      }*/

      const botMessage = {
        id: Date.now() + 1,
        type: "bot",
        text: botText.trim()
      };

      setMessages(prev => [...prev, botMessage]);

    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          id: Date.now() + 1,
          type: "bot",
          text: "❌ Could not reach the backend: " + err.message
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
