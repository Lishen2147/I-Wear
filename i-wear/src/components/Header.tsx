"use client";

import React, { useState } from "react";

const Header: React.FC = () => {
  return (
    <header className="h-18 bg-white grid sm:grid-cols-1 md:grid-cols-3 lg:grid-cols-5 py-4 px-6 space-x-8 border-b shadow-md z-50 select-none">
      <div className="relative flex items-center space-x-3 xl:flex-grow cursor-pointer text-xl text-black">
        I-Wear
      </div>
    </header>
  );
};

export default Header;
