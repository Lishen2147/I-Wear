"use client";

import Image from "next/image";

interface PredictionImagesProps {
  images: string[] | null;
  probabilities: number[] | null;
}

const glasses_dict = {
  heart: ["oval", "rectangle", "round"],
  oblong: ["horn", "oval", "square"],
  oval: ["oval", "rectangle", "round"],
  round: ["oval", "rectangle", "square"],
  square: ["large", "oval", "round"],
};

const PredictionImages: React.FC<PredictionImagesProps> = ({
  images,
  probabilities,
}) => {
  const topGlasses =
    glasses_dict[
      probabilities?.sort((a, b) => b[1] - a[1])[0][0].toLowerCase()
    ];

  return (
    <div className="flex gap-4">
      {images &&
        images?.map((img, idx) => (
          <div className="relative group">
            <Image
              key={idx}
              className="drop-shadow-lg brightness-[0.45] transition-all duration-[800ms] group-hover:brightness-100"
              src={`data:image/png;base64,${img}`}
              alt="Generated Image"
              width={450}
              height={450}
            />
            <div className="flex flex-col items-center gap-2 absolute top-1/2 left-1/2 translate-x-[-50%] translate-y-[-50%] group-hover:invisible">
              <div className="text-3xl font-bold text-white">{`Option ${idx + 1}`}</div>
              <div className="text-xl text-white capitalize">
                {topGlasses && topGlasses[idx]}
              </div>
            </div>
          </div>
        ))}
    </div>
  );
};

export default PredictionImages;
