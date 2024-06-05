"use client";

import React from "react";
import { FaHeart } from "react-icons/fa";

const Header: React.FC = () => {
  return (
    <header className="flex flex-row items-center justify-between p-4 bg-gradient-to-br from-blue-400 via-sky-400 to-cyan-400 w-full">
      <div className="font-semibold font-sans text-4xl text-white">
      👓 I-Wear
      </div>
      <FaHeart className="text-red-500 text-4xl transition-transform duration-500 hover:scale-125" />
    </header>
  );
};

export default Header;
