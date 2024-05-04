"use client";

import React from "react";

interface LoadingProps {
    isLoading: boolean,
    faceStructure: string | null
}

const Loading: React.FC<LoadingProps> = ({isLoading, faceStructure}) => {

  return (
    <div>
        {isLoading ? <div>Loading...</div> : <div>{"FACIAL STRUCTURE: " + faceStructure}</div>}
    </div>
  )
};

export default Loading;
