// src/components/FARSChatbot.jsx
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
    "Show me accidents involving teenage drivers in rainy conditions that resulted in a fatality."
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

  const handleSend = () => {
    if (!inputValue.trim() || isDisabled) return

    const userMessage = {
      id: Date.now(),
      type: "user",
      text: inputValue.trim()
    }

    // append instead of overwrite
    setMessages(prev => [...prev, userMessage])
    setInputValue("")
    setIsDisabled(true)
    setIsThinking(true)

    // TODO: replace this timeout with a real call to your RAG backend
    setTimeout(() => {
      setIsThinking(false)
      setIsDisabled(false)

      const botMessage = {
        id: Date.now() + 1,
        type: "bot",
        text:
          "Based on the provided reports, one relevant incident is ST_CASE 12345. " +
          "This was a fatal accident in the rain involving a 17-year-old driver in a 2018 Honda Civic. " +
          "The report indicates the injury severity for the driver was fatal."
      }

      setMessages(prev => [...prev, botMessage])
    }, 1500)
  }

  const handleKeyDown = e => {
    if (e.key === "Enter" && !e.shiftKey && !isDisabled) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div
      className="min-h-screen flex items-center justify-center bg-[#f4f4f4] p-4"
      style={{ fontFamily: "'Montserrat', sans-serif" }}
    >
      <style>
        {`
          @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
          
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

      <div className="w-full max-w-2xl bg-[#ffffff] rounded-2xl shadow-lg overflow-hidden">
        <div className="bg-[#630031] px-6 py-4">
          <h1 className="text-[#ffffff] text-xl font-semibold">
            FARS Conversational Query
          </h1>
        </div>

        <div className="h-96 overflow-y-auto p-6 space-y-4">
          {messages.map(message => (
            <div
              key={message.id}
              className={`flex ${
                message.type === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-[80%] px-4 py-3 rounded-2xl ${
                  message.type === "user"
                    ? "bg-[#CF5A00] text-[#ffffff]"
                    : "bg-[#630031] text-[#ffffff]"
                }`}
              >
                <p className="text-sm leading-relaxed">{message.text}</p>
              </div>
            </div>
          ))}

          {isThinking && (
            <div className="flex justify-start">
              <div className="max-w-[80%] px-4 py-3 rounded-2xl bg-[#630031] text-[#ffffff]">
                <p className="text-sm thinking-dots">Analyzing data</p>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className="border-t border-[#f4f4f4] p-4">
          <div className="flex gap-3">
            <input
              type="text"
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isDisabled}
              className="flex-1 px-4 py-3 border-2 border-[#630031] rounded-full text-[#333333] focus:outline-none focus:border-[#CF5A00] disabled:bg-[#f4f4f4] disabled:cursor-not-allowed text-sm"
              placeholder="Type your query..."
            />
            <button
              onClick={handleSend}
              disabled={isDisabled}
              className="px-6 py-3 bg-[#CF5A00] text-[#ffffff] rounded-full font-semibold hover:bg-[#b34f00] disabled:bg-[#cccccc] disabled:cursor-not-allowed transition-colors text-sm"
            >
              Send
            </button>
          </div>
          <p className="text-center text-xs text-[#999999] mt-3">
            Supported by the Fatality Analysis Reporting System (FARS)
          </p>
        </div>
      </div>
    </div>
  )
}
