"use client";

import React from "react";
import Loading from "./Loading";

interface ResponseProps {
    isLoading: boolean,
    isRequesting: boolean,
    faceStructure: string | null
}

const Response: React.FC<ResponseProps> = ({isLoading, isRequesting, faceStructure}) => {

  return (
    <div>
        {!isRequesting ? <div>INPUT IMAGE FOR FACIAL SHAPE</div> : <Loading isLoading={isLoading} faceStructure={faceStructure} />}
    </div>
  )
};

export default Response;
