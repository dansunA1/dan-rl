#[macro_export]
macro_rules! single_match {
    (($var:expr => $target:pat in [ $( $pat:pat => $val:expr ;)* ]), $body:expr) => {
        match $var {
            $(
                $pat => {
                    let $target = $val;
                    $body
                }
            ),*
        }
    };
}

#[macro_export]
macro_rules! cartesian_map {
    // Base case: empty list
    ([], $body:expr) => {
        $body
    };

    // Recursive case: list = head, tail...
    (
        [ $head:tt $(, $rest:tt)* $(,)? ],
        $body:expr
    ) => {
        single_match!(
            $head,
            cartesian_map!([ $( $rest ),* ], $body)
        )
    };
}