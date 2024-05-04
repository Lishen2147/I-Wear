"use client";
import React, { ReactElement } from "react";
import Header from "../components/Header";
import Upload from "../components/Upload";


export default function Home() {

  return (
    <main className="bg-black w-screen h-screen">
      <Header />
      <Upload />
    </main>
  );
}
